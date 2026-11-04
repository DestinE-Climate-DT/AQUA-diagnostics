# ruff: noqa: N999
from .ensembleMaps import EnsembleMaps


class EnsembleProfiles(EnsembleMaps):
    """
    Compute mean and standard deviation of 1D ensembles such as latitude profiles and seasonal cycles.

    The statistics are the same as in `EnsembleMaps`: only the ensemble dimension is reduced,
    so any other dimension of the input (e.g. lat, month, period) is preserved.
    """

    def __init__(self, diagnostic_product="EnsembleProfiles", **kwargs):
        """
        Initialize the EnsembleProfiles class.

        Args:
            diagnostic_product (str, optional): Product name used in the output filenames.
                Defaults to "EnsembleProfiles".
            **kwargs: Keyword arguments of `EnsembleMaps`.
        """
        super().__init__(diagnostic_product=diagnostic_product, **kwargs)
