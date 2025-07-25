"""Kinetic modeling interfaces and classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

from .. import __version__
from ..utils.kinmod import (
    load_blood,
    load_tacs,
    save_kinpar_json,
    save_kinpar_tsv,
)


class BaseBloodModel:
    """Base class for kinetic models using blood input."""

    parameters: list[str] = []

    def __init__(
        self,
        tac_times: np.ndarray,
        tac_values: np.ndarray,
        plasma_times: np.ndarray,
        plasma_values: np.ndarray,
        blood_values: np.ndarray | None = None,
        n_iterations: int = 50,
    ) -> None:
        self.tac_times = tac_times / 60.0
        self.tac_values = tac_values
        self.plasma_times = plasma_times / 60.0
        self.plasma_values = plasma_values
        self.blood_values = blood_values if blood_values is not None else plasma_values
        self.n_iterations = n_iterations

    def fit(self) -> dict:
        raise NotImplementedError

    def visualize_fit(self, output_path: str, region_name: str) -> None:
        raise NotImplementedError


class MA1Model(BaseBloodModel):
    parameters = [
        'VT',
        'intercept',
        'coef_X2',
        'MSE',
        'SigmaSqr',
        'LogLike',
        'AIC',
        'FPE',
        'CoV',
    ]

    def __init__(
        self,
        tac_times: np.ndarray,
        tac_values: np.ndarray,
        plasma_times: np.ndarray,
        plasma_values: np.ndarray,
        t_star: float,
        **kwargs,
    ) -> None:
        super().__init__(tac_times, tac_values, plasma_times, plasma_values, **kwargs)
        self.t_star = t_star

    def fit(self) -> dict:
        import statsmodels.api as sm
        from scipy.integrate import cumtrapz

        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        auc_input = cumtrapz(plasma_interp, tac_minutes, initial=0)
        auc_pet = cumtrapz(self.tac_values, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star
        X = np.column_stack((auc_input[mask], auc_pet[mask]))
        Y = self.tac_values[mask]

        model = sm.OLS(Y, X)
        results = model.fit()

        b1, b2 = results.params
        y_pred = results.fittedvalues
        residuals = Y - y_pred
        n = len(Y)
        p = 2

        mean_y = np.mean(Y)
        cov = np.std(residuals, ddof=p) / mean_y if mean_y != 0 else np.nan
        mse = np.sum(residuals**2) / (n - p)
        sigma_squared = np.var(residuals, ddof=p)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - 0.5 * np.sum(
            residuals**2
        ) / sigma_squared
        aic = -2 * log_likelihood + 2 * (p + 1)
        fpe = np.sum(residuals**2) * (n + p) / (n - p)

        VT = -b1 / b2 if b2 != 0 else np.nan
        intercept = 1.0 / b2 if b2 != 0 else np.nan

        self.fit_result = {
            'VT': VT,
            'intercept': intercept,
            'coef_X2': b1,
            'MSE': mse,
            'SigmaSqr': sigma_squared,
            'LogLike': log_likelihood,
            'AIC': aic,
            'FPE': fpe,
            'CoV': cov,
        }
        return self.fit_result

    def visualize_fit(self, output_path: str, region_name: str) -> None:
        import matplotlib.pyplot as plt
        from scipy.integrate import cumtrapz

        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        auc_input = cumtrapz(plasma_interp, tac_minutes, initial=0)
        auc_pet = cumtrapz(self.tac_values, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star

        b1 = self.fit_result['coef_X2']
        b2 = -b1 / self.fit_result['VT'] if self.fit_result['VT'] != 0 else np.nan
        fit_line = b1 * auc_input + b2 * auc_pet

        plt.figure(figsize=(8, 4))
        plt.plot(tac_minutes, self.tac_values, 'ko', label='TAC')
        plt.plot(tac_minutes[mask], self.tac_values[mask], 'ro', label='Fitting Points')
        plt.plot(tac_minutes, fit_line, 'r--', label='MA1 Fit')
        plt.title(region_name)
        plt.xlabel('Time (min)')
        plt.ylabel('Radioactivity Concentration')
        plt.legend()
        plt.annotate(
            (
                f"VT = {self.fit_result['VT']:.2f}\n"
                f"Intercept = {self.fit_result['intercept']:.2f}\n"
                f"CoV = {self.fit_result['CoV']:.4f}"
            ),
            xy=(0.6, 0.1),
            xycoords='axes fraction',
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class LoganModel(BaseBloodModel):
    parameters = [
        'VT',
        'Kappa2',
        'VT_var',
        'intercept',
        'R_squared',
        'MSE',
        'SigmaSqr',
        'LogLike',
        'AIC',
        'FPE',
        'CoV',
    ]

    def __init__(
        self,
        tac_times: np.ndarray,
        tac_values: np.ndarray,
        plasma_times: np.ndarray,
        plasma_values: np.ndarray,
        blood_values: np.ndarray | None = None,
        t_star: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tac_times,
            tac_values,
            plasma_times,
            plasma_values,
            blood_values=blood_values,
            **kwargs,
        )
        self.t_star = t_star if t_star is not None else tac_times[0]

    def fit(self) -> dict:
        import statsmodels.api as sm
        from scipy.integrate import cumtrapz

        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        integral_tac = cumtrapz(self.tac_values, tac_minutes, initial=0)
        integral_plasma = cumtrapz(plasma_interp, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star
        x = integral_plasma[mask] / self.tac_values[mask]
        y = integral_tac[mask] / self.tac_values[mask]

        X_design = sm.add_constant(x)
        glm_model = sm.WLS(y, X_design)
        glm_results = glm_model.fit()

        intercept, VT = glm_results.params
        cov_matrix = glm_results.cov_params()
        VT_var = cov_matrix[1, 1]

        y_pred = glm_results.fittedvalues
        residuals = y - y_pred
        n = len(y)
        p = 2

        mean_y = np.mean(y)
        cov = np.std(residuals, ddof=p) / mean_y if mean_y != 0 else np.nan
        mse = np.sum(residuals**2) / (n - p)
        sigma_squared = np.var(residuals, ddof=p)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - 0.5 * np.sum(
            residuals**2
        ) / sigma_squared
        aic = -2 * log_likelihood + 2 * (p + 1)
        fpe = np.sum(residuals**2) * (n + p) / (n - p)
        r_squared = glm_results.rsquared

        Kappa2 = -1 / intercept if intercept != 0 else np.nan

        self.fit_result = {
            'VT': VT,
            'Kappa2': Kappa2,
            'VT_var': VT_var,
            'intercept': intercept,
            'R_squared': r_squared,
            'MSE': mse,
            'SigmaSqr': sigma_squared,
            'LogLike': log_likelihood,
            'AIC': aic,
            'FPE': fpe,
            'CoV': cov,
        }
        return self.fit_result

    def visualize_fit(self, output_path: str, region_name: str) -> None:
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        from scipy.integrate import cumtrapz

        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        integral_tac = cumtrapz(self.tac_values, tac_minutes, initial=0)
        integral_plasma = cumtrapz(plasma_interp, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star
        x = integral_plasma[mask] / self.tac_values[mask]
        y = integral_tac[mask] / self.tac_values[mask]

        X_design = sm.add_constant(x)
        y_pred = X_design @ np.array([self.fit_result['intercept'], self.fit_result['VT']])

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'ko', label='Data')
        plt.plot(x, y_pred, 'r--', label='Logan Fit')
        plt.xlabel('∫Cp(t)/Ct(t) [min]')
        plt.ylabel('∫Ct(t)/Ct(t) [min]')
        plt.title(f'Logan Plot - {region_name}')
        plt.annotate(
            (
                f"VT = {self.fit_result['VT']:.2f}\n"
                f"Kappa2 = {self.fit_result['Kappa2']:.4f}\n"
                f"t_star = {self.t_star:.2f} min\n"
                f"CoV = {self.fit_result['CoV']:.4f}"
            ),
            xy=(0.65, 0.1),
            xycoords='axes fraction',
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class TwoTCMModel(BaseBloodModel):
    parameters = ['K1', 'k2', 'k3', 'k4', 'vB', 'VT', 'CoV']

    def __init__(
        self,
        tac_times: np.ndarray,
        tac_values: np.ndarray,
        plasma_times: np.ndarray,
        plasma_values: np.ndarray,
        blood_values: np.ndarray | None = None,
        bounds_lower: list[float] | None = None,
        bounds_upper: list[float] | None = None,
        vB_fixed: float | None = None,
        inpshift: float = 0.0,
        fit_end_time: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tac_times,
            tac_values,
            plasma_times,
            plasma_values,
            blood_values=blood_values,
            **kwargs,
        )
        self.vB_fixed = vB_fixed
        self.inpshift = inpshift
        self.fit_end_time = fit_end_time or self.tac_times[-1]

        if self.vB_fixed is not None:
            self.bounds_lower = bounds_lower or [0.0001, 0.0001, 0.0001, 0.0001]
            self.bounds_upper = bounds_upper or [1.0, 0.5, 0.5, 0.5]
        else:
            self.bounds_lower = bounds_lower or [0.001, 0.001, 0.001, 0.001, 0.01]
            self.bounds_upper = bounds_upper or [1.0, 0.5, 0.5, 0.5, 0.1]

    def fit(self) -> dict:
        from scipy.interpolate import interp1d
        from scipy.optimize import least_squares

        mask = self.tac_times <= self.fit_end_time
        t_pet = self.tac_times[mask]
        tac_pet = self.tac_values[mask]

        plasma_shifted_times = self.plasma_times + self.inpshift
        Cp = interp1d(
            plasma_shifted_times,
            self.plasma_values,
            bounds_error=False,
            fill_value='extrapolate',
        )(t_pet)
        Cb = interp1d(
            plasma_shifted_times,
            self.blood_values,
            bounds_error=False,
            fill_value='extrapolate',
        )(t_pet)

        best_fit = None
        min_cost = np.inf

        for _ in range(self.n_iterations):
            x0 = np.random.uniform(self.bounds_lower, self.bounds_upper)
            res = least_squares(
                self._residuals,
                x0,
                bounds=(self.bounds_lower, self.bounds_upper),
                args=(t_pet, Cp, Cb, tac_pet),
            )
            if res.cost < min_cost:
                min_cost = res.cost
                best_fit = res.x

        if self.vB_fixed is not None:
            K1, k2, k3, k4 = best_fit
            vB = self.vB_fixed
        else:
            K1, k2, k3, k4, vB = best_fit

        VT = (K1 / k2) * (1 + k3 / k4)

        residuals = self._residuals(best_fit, t_pet, Cp, Cb, tac_pet)
        mean_tac_pet = np.mean(tac_pet)
        cov = np.std(residuals, ddof=len(best_fit)) / mean_tac_pet if mean_tac_pet != 0 else np.nan

        self.fit_result = {
            'K1': K1,
            'k2': k2,
            'k3': k3,
            'k4': k4,
            'vB': vB,
            'VT': VT,
            'CoV': cov,
        }

        return self.fit_result

    def _residuals(self, params, t, Cp, Cb, tac_pet):

        if self.vB_fixed is not None:
            K1, k2, k3, k4 = params
            vB = self.vB_fixed
        else:
            K1, k2, k3, k4, vB = params

        Ct_model = self._simulate_2tcm(t, Cp, Cb, K1, k2, k3, k4, vB)
        return Ct_model - tac_pet

    def _simulate_2tcm(self, t, Cp, Cb, K1, k2, k3, k4, vB):
        dt = np.diff(t, prepend=0)
        C1, C2, Ct = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

        for i in range(1, len(t)):
            dC1 = dt[i] * (K1 * Cp[i] - (k2 + k3) * C1[i - 1] + k4 * C2[i - 1])
            dC2 = dt[i] * (k3 * C1[i - 1] - k4 * C2[i - 1])
            C1[i] = C1[i - 1] + dC1
            C2[i] = C2[i - 1] + dC2
            Ct[i] = (1 - vB) * (C1[i] + C2[i]) + vB * Cb[i]

        return Ct

    def visualize_fit(self, output_path: str, region_name: str) -> None:
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        mask = self.tac_times <= self.fit_end_time
        t_pet = self.tac_times[mask]

        plasma_shifted_times = self.plasma_times + self.inpshift
        Cp = interp1d(
            plasma_shifted_times,
            self.plasma_values,
            bounds_error=False,
            fill_value='extrapolate',
        )(t_pet)
        Cb = interp1d(
            plasma_shifted_times,
            self.blood_values,
            bounds_error=False,
            fill_value='extrapolate',
        )(t_pet)

        fit_curve = self._simulate_2tcm(
            t_pet,
            Cp,
            Cb,
            K1=self.fit_result['K1'],
            k2=self.fit_result['k2'],
            k3=self.fit_result['k3'],
            k4=self.fit_result['k4'],
            vB=self.fit_result['vB'],
        )

        plt.figure(figsize=(8, 4))
        plt.plot(t_pet, self.tac_values[mask], 'ko', label='Measured TAC')
        plt.plot(t_pet, fit_curve, 'r-', label='2TCM Fit')
        plt.title(region_name)
        plt.xlabel('Time (min)')
        plt.ylabel('Radioactivity Concentration')
        plt.annotate(
            (
                f"VT = {self.fit_result['VT']:.2f}\n"
                f"CoV = {self.fit_result['CoV']:.4f}\n"
                f"vB = {self.fit_result['vB']:.4f}"
            ),
            xy=(0.65, 0.5),
            xycoords='axes fraction',
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class OneTCMModel(BaseBloodModel):
    parameters = ['K1', 'k2', 'vB', 'VT', 'CoV']

    def __init__(
        self,
        tac_times: np.ndarray,
        tac_values: np.ndarray,
        plasma_times: np.ndarray,
        plasma_values: np.ndarray,
        blood_values: np.ndarray | None = None,
        bounds_lower: list[float] | None = None,
        bounds_upper: list[float] | None = None,
        vB_fixed: float | None = None,
        fit_end_time: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tac_times,
            tac_values,
            plasma_times,
            plasma_values,
            blood_values=blood_values,
            **kwargs,
        )
        self.bounds_lower = bounds_lower or [0.0001, 0.0001, 0.01]
        self.bounds_upper = bounds_upper or [1.0, 0.5, 0.1]
        self.vB_fixed = vB_fixed
        self.fit_end_time = fit_end_time or self.tac_times[-1]

    def fit(self) -> dict:
        from scipy.interpolate import interp1d
        from scipy.optimize import least_squares

        mask = self.tac_times <= self.fit_end_time
        tac_minutes = self.tac_times[mask]
        tac_values = self.tac_values[mask]

        plasma_minutes = self.plasma_times
        ca = interp1d(plasma_minutes, self.plasma_values, fill_value='extrapolate')
        cb = interp1d(plasma_minutes, self.blood_values, fill_value='extrapolate')

        def residuals(params):
            if self.vB_fixed is not None:
                K1, k2 = params
                vB = self.vB_fixed
            else:
                K1, k2, vB = params
            Ct_pred = self._simulate_1tcm(tac_minutes, ca, cb, K1, k2, vB)
            return Ct_pred - tac_values

        best_fit = None
        min_cost = np.inf

        if self.vB_fixed is not None:
            bounds_lower = self.bounds_lower[:2]
            bounds_upper = self.bounds_upper[:2]
        else:
            bounds_lower = self.bounds_lower
            bounds_upper = self.bounds_upper

        for _ in range(self.n_iterations):
            x0 = np.random.uniform(bounds_lower, bounds_upper)
            res = least_squares(residuals, x0, bounds=(bounds_lower, bounds_upper))
            if res.cost < min_cost:
                min_cost = res.cost
                best_fit = res.x

        if self.vB_fixed is not None:
            K1, k2 = best_fit
            vB = self.vB_fixed
        else:
            K1, k2, vB = best_fit

        VT = K1 / k2 if k2 != 0 else np.nan

        residual_values = residuals(best_fit)
        mean_tac_pet = np.mean(tac_values)
        cov = (
            np.std(residual_values, ddof=len(best_fit)) / mean_tac_pet
            if mean_tac_pet != 0
            else np.nan
        )

        self.fit_result = {'K1': K1, 'k2': k2, 'vB': vB, 'VT': VT, 'CoV': cov}
        return self.fit_result

    def visualize_fit(self, output_path: str, region_name: str) -> None:
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        mask = self.tac_times <= self.fit_end_time
        tac_minutes = self.tac_times[mask]

        plasma_minutes = self.plasma_times
        ca = interp1d(plasma_minutes, self.plasma_values, fill_value='extrapolate')
        cb = interp1d(plasma_minutes, self.blood_values, fill_value='extrapolate')

        fit_curve = self._simulate_1tcm(
            tac_minutes,
            ca,
            cb,
            K1=self.fit_result['K1'],
            k2=self.fit_result['k2'],
            vB=self.fit_result['vB'],
        )

        plt.figure(figsize=(8, 4))
        plt.plot(tac_minutes, self.tac_values[mask], 'ko', label='Measured TAC')
        plt.plot(tac_minutes, fit_curve, 'r--', label='1TCM Fit')
        plt.title(region_name)
        plt.xlabel('Time (min)')
        plt.ylabel('Radioactivity Concentration')
        plt.annotate(
            (
                f"VT = {self.fit_result['VT']:.2f}\n"
                f"CoV = {self.fit_result['CoV']:.4f}\n"
                f"vB = {self.fit_result['vB']:.4f}"
            ),
            xy=(0.65, 0.5),
            xycoords='axes fraction',
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _simulate_1tcm(self, t, ca_func, cb_func, K1, k2, vB):
        dt = np.diff(t, prepend=0)
        Ct = np.zeros_like(t)
        C1 = 0

        for i in range(1, len(t)):
            dC1 = dt[i] * (K1 * ca_func(t[i]) - k2 * C1)
            C1 += dC1
            Ct[i] = (1 - vB) * C1 + vB * cb_func(t[i])

        return Ct


class _FitKMInputSpec(BaseInterfaceInputSpec):
    tacs_file = File(exists=True, mandatory=True, desc='Regional TACs TSV file')
    blood_file = File(exists=True, mandatory=True, desc='Blood data TSV file')
    model = traits.Enum('logan', 'ma1', '1tcm', '2tcm', mandatory=True, desc='Model name')
    t_star = traits.Float(desc='t* for linear models')
    vB_fixed = traits.Float(desc='Fixed blood volume fraction')
    fit_end_time = traits.Float(desc='End time for fitting in minutes')
    n_iterations = traits.Int(50, usedefault=True, desc='Number of optimization iterations')
    save_figures = traits.Bool(False, usedefault=True, desc='Save fit plots')


class _FitKMOutputSpec(TraitedSpec):
    params_file = File(exists=True, desc='Kinetic parameters TSV')
    metadata_file = File(exists=True, desc='Kinetic parameters JSON metadata')


class FitKineticModel(SimpleInterface):
    """Fit kinetic models to regional TACs using blood data."""

    input_spec = _FitKMInputSpec
    output_spec = _FitKMOutputSpec

    def _run_interface(self, runtime):
        tacs = load_tacs(self.inputs.tacs_file)
        blood = load_blood(self.inputs.blood_file)

        times = 0.5 * (tacs['FrameTimesStart'] + tacs['FrameTimesEnd'])
        plasma_times = blood['time'].to_numpy()
        plasma_values = blood['plasma_radioactivity'].to_numpy()
        blood_values = blood['whole_blood_radioactivity'].to_numpy()

        model_name = self.inputs.model
        ModelClass = {
            'ma1': MA1Model,
            'logan': LoganModel,
            '1tcm': OneTCMModel,
            '2tcm': TwoTCMModel,
        }[model_name]

        results = []
        for region in tacs.columns[2:]:
            model_kwargs = {
                'tac_times': times.to_numpy(),
                'tac_values': tacs[region].to_numpy(),
                'plasma_times': plasma_times,
                'plasma_values': plasma_values,
                'blood_values': blood_values,
                'n_iterations': self.inputs.n_iterations,
            }
            if model_name in {'ma1', 'logan'} and isdefined(self.inputs.t_star):
                model_kwargs['t_star'] = (
                    self.inputs.t_star / 60.0
                    if model_name == 'logan'
                    else self.inputs.t_star
                )
            if model_name in {'1tcm', '2tcm'}:
                if isdefined(self.inputs.vB_fixed):
                    model_kwargs['vB_fixed'] = self.inputs.vB_fixed
                if isdefined(self.inputs.fit_end_time):
                    model_kwargs['fit_end_time'] = self.inputs.fit_end_time
            model = ModelClass(**model_kwargs)
            res = model.fit()
            if self.inputs.save_figures:
                figfile = runtime.cwd / f'{region}_{model_name}.png'
                model.visualize_fit(str(figfile), region)
            results.append({'name': region, **res})

        out_tsv = Path(runtime.cwd) / 'kinpar.tsv'
        df = pd.DataFrame(results)
        df_path = save_kinpar_tsv(df, out_tsv)
        meta = {
            'Description': f'{model_name} kinetic modeling results',
            'ModelName': model_name,
            'BloodType': 'arterial',
            'SoftwareName': 'petprep',
            'SoftwareVersion': __version__,
            'Parameters': ModelClass.parameters,
        }
        meta_path = save_kinpar_json(meta, Path(runtime.cwd) / 'kinpar.json')

        self._results['params_file'] = str(df_path)
        self._results['metadata_file'] = str(meta_path)

        return runtime


__all__ = [
    'FitKineticModel',
    'BaseBloodModel',
    'MA1Model',
    'LoganModel',
    'OneTCMModel',
    'TwoTCMModel',
]
