from nipype.interfaces.freesurfer import GTMSeg as _GTMSeg


class GTMSeg(_GTMSeg):
    """Thin wrapper around Nipype's :class:`~nipype.interfaces.freesurfer.GTMSeg`."""


__all__ = ('GTMSeg',)
