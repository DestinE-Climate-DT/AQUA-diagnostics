# ruff: noqa: N999
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.core.logger import log_configure

from .base import BaseMixin
from .util import compute_statistics

xr.set_options(keep_attrs=True)


class EnsembleZonal(BaseMixin):
    """
    Compute mean and standard deviation of zonal average ensembles.

    This class takes an ensemble dataset containing 2D zonal data (level-latitude)
    and computes the mean and standard deviation across the specified ensemble
    dimension. Ensure that the dataset has the correct spatial dimensions
    (e.g., lev and lat) before computing statistics.
    """

    def __init__(
        self,
        var=None,
        dataset=None,
        catalog_list=None,
        model_list=None,
        exp_list=None,
        source_list=None,
        ensemble_dimension_name="ensemble",
        outputdir="./",
        loglevel="WARNING",
    ):
        """
        Initialize the EnsembleZonal class.

        Args:
            var (str, optional): Variable name to compute statistics for. Defaults to None.
            dataset (xr.Dataset, optional): Dataset of 2D (level-latitude) ensemble members,
                concatenated along the ensemble dimension. Defaults to None.
            catalog_list (list[str], optional): List of catalog names. Defaults to None.
            model_list (list[str], optional): List of model names. Defaults to None.
            exp_list (list[str], optional): List of experiment names. Defaults to None.
            source_list (list[str], optional): List of source names. Defaults to None.
            ensemble_dimension_name (str, optional): Name of the dimension along which individual
                datasets are concatenated. Defaults to "ensemble".
            outputdir (str, optional): Output directory path for saving files. Defaults to "./".
            loglevel (str, optional): Logging level. Defaults to "WARNING".
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=self.loglevel, log_name="Ensemble Zonal Averages")

        self.var = var
        self.dataset = dataset
        self.dim = ensemble_dimension_name
        self.dataset_mean = None
        self.dataset_std = None
        self.outputdir = outputdir

        super().__init__(
            diagnostic_product="EnsembleZonal",
            catalog_list=catalog_list,
            model_list=model_list,
            exp_list=exp_list,
            source_list=source_list,
            loglevel=loglevel,
            outputdir=self.outputdir,
        )

    def run(self):
        """
        Compute the mean and standard deviation of the input dataset.

        It is important to ensure that the dimension along which the statistics
        are computed (defined by `ensemble_dimension_name`, default: "ensemble")
        matches the input data. Once computed, the mean and standard deviation
        are automatically saved to NetCDF files.

        Raises:
            NoDataError: If no dataset was provided during initialization.
        """
        self.logger.info("Compute function in EnsembleZonal")

        if self.dataset is not None:
            self.dataset_mean, self.dataset_std = compute_statistics(
                variable=self.var, ds=self.dataset, ens_dim=self.dim, loglevel=self.loglevel
            )
            self.save_netcdf(
                var=self.var,
                data_name="mean",
                data=self.dataset_mean,
            )
            self.save_netcdf(
                var=self.var,
                data_name="std",
                data=self.dataset_std,
            )
        else:
            self.logger.info("No ensemble data is provided to the compute method")
            raise NoDataError("No data is given to the compute method")
