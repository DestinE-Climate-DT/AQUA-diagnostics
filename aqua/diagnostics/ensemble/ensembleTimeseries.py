# ruff: noqa: N999
import xarray as xr

from aqua.core.logger import log_configure

# from aqua.exceptions import NoDataError
from .base import BaseMixin
from .util import compute_statistics

xr.set_options(keep_attrs=True)


class EnsembleTimeseries(BaseMixin):
    """
    Compute mean and standard deviation of timeseries ensembles.

    This class takes hourly, daily, monthly, or annual timeseries data from
    multiple ensemble members, computes their point-wise mean and standard
    deviation along the specified ensemble dimension, and optionally saves
    the results to NetCDF files.

    Note:
        The standard deviation (STD) is computed point-wise along the mean.
    """
    def __init__(
        self,
        var=None,
        hourly_data=None,
        daily_data=None,
        monthly_data=None,
        annual_data=None,
        catalog_list=None,
        model_list=None,
        exp_list=None,
        source_list=None,
        ensemble_dimension_name="ensemble",
        description=None,
        outputdir="./",
        loglevel="WARNING",
    ):
        """
        Initialize the EnsembleTimeseries class.

        Args:
            var (str, optional): Variable name. Defaults to None.
            hourly_data (xr.Dataset, optional): Dataset of hourly timeseries ensemble members,
                concatenated along the ensemble dimension. Defaults to None.
            daily_data (xr.Dataset, optional): Dataset of daily timeseries ensemble members,
                concatenated along the ensemble dimension. Defaults to None.
            monthly_data (xr.Dataset, optional): Dataset of monthly timeseries ensemble members,
                concatenated along the ensemble dimension. Defaults to None.
            annual_data (xr.Dataset, optional): Dataset of annual timeseries ensemble members,
                concatenated along the ensemble dimension. Defaults to None.
            catalog_list (list[str], optional): List of catalog names. Defaults to None.
            model_list (list[str], optional): List of model names. This is mandatory for saving. Defaults to None.
            exp_list (list[str], optional): List of experiment names. Defaults to None.
            source_list (list[str], optional): List of source names. Defaults to None.
            ensemble_dimension_name (str, optional): Name of the dimension along which individual
                datasets are concatenated. Defaults to "ensemble".
            description (str, optional): Description to include in the output NetCDF metadata. Defaults to None.
            outputdir (str, optional): Output directory path for saving files. Defaults to "./".
            loglevel (str, optional): Logging level. Defaults to "WARNING".
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=self.loglevel, log_name="Ensemble Timeseries")
        self.var = var
        self.dim = ensemble_dimension_name
        self.diagnostic_product = "EnsembleTimeseries"

        self.hourly_data = hourly_data
        self.daily_data = daily_data
        self.monthly_data = monthly_data
        self.annual_data = annual_data

        self.catalog_list = catalog_list
        self.model_list = model_list
        self.exp_list = exp_list
        self.source_list = source_list

        self.hourly_data_mean = None
        self.hourly_data_std = None

        self.daily_data_mean = None
        self.daily_data_std = None

        self.monthly_data_mean = None
        self.monthly_data_std = None

        self.annual_data_mean = None
        self.annual_data_std = None

        self.description = description
        self.outputdir = outputdir

        super().__init__(
            diagnostic_product="EnsembleTimeseries",
            catalog_list=catalog_list,
            model_list=model_list,
            exp_list=exp_list,
            source_list=source_list,
            loglevel=loglevel,
            outputdir=self.outputdir,
        )

    def run(self):
        """
        Compute the mean and standard deviation for the provided datasets.

        It is important to ensure that the dimension along which the mean is
        computed (defined by `ensemble_dimension_name`, default: "ensemble")
        correctly matches the input data. Once computed, the statistics (mean
        and STD) for each provided frequency (hourly, daily, monthly, annual)
        are automatically saved to NetCDF files.

        TODO:
            - Test Dask's `.compute()` function within this execution flow.
        """
        self.logger.info("Compute function in EnsembleTimeseries")

        # For Hourly data
        if self.hourly_data is not None:
            self.hourly_data_mean, self.hourly_data_std = compute_statistics(
                variable=self.var, ds=self.hourly_data, ens_dim=self.dim, loglevel=self.loglevel
            )
            self.save_netcdf(
                var=self.var,
                freq="hourly",
                data_name="mean",
                data=self.hourly_data_mean,
                description=self.description,
                startdate=self.hourly_data_mean.time.values[0],
                enddate=self.hourly_data_mean.time.values[-1],
            )
            self.save_netcdf(
                var=self.var,
                freq="hourly",
                data_name="std",
                data=self.hourly_data_std,
                description=self.description,
                startdate=self.hourly_data_std.time.values[0],
                enddate=self.hourly_data_std.time.values[-1],
            )
        else:
            self.logger.info("No hourly ensemble data is provided")

        # For Daily data
        if self.daily_data is not None:
            self.daily_data_mean, self.daily_data_std = compute_statistics(
                variable=self.var, ds=self.daily_data, ens_dim=self.dim, loglevel=self.loglevel
            )
            self.save_netcdf(
                var=self.var,
                freq="daily",
                data_name="mean",
                data=self.daily_data_mean,
                description=self.description,
                startdate=self.daily_data_mean.time.values[0],
                enddate=self.daily_data_mean.time.values[-1],
            )
            self.save_netcdf(
                var=self.var,
                freq="daily",
                data_name="std",
                data=self.daily_data_std,
                description=self.description,
                startdate=self.daily_data_std.time.values[0],
                enddate=self.daily_data_std.time.values[-1],
            )
        else:
            self.logger.info("No daily ensemble data is provided")

        # For Monthly data
        if self.monthly_data is not None:
            self.monthly_data_mean, self.monthly_data_std = compute_statistics(
                variable=self.var, ds=self.monthly_data, ens_dim=self.dim, loglevel=self.loglevel
            )
            self.save_netcdf(
                var=self.var,
                freq="monthly",
                data_name="mean",
                data=self.monthly_data_mean,
                description=self.description,
                startdate=self.monthly_data_mean.time.values[0],
                enddate=self.monthly_data_mean.time.values[-1],
            )
            self.save_netcdf(
                var=self.var,
                freq="monthly",
                data_name="std",
                data=self.monthly_data_std,
                description=self.description,
                startdate=self.monthly_data_std.time.values[0],
                enddate=self.monthly_data_std.time.values[-1],
            )
        else:
            self.logger.info("No monthly ensemble data is provided")

        # For Annual data
        if self.annual_data is not None:
            self.annual_data_mean, self.annual_data_std = compute_statistics(
                variable=self.var, ds=self.annual_data, ens_dim=self.dim, loglevel=self.loglevel
            )
            self.save_netcdf(
                var=self.var,
                freq="annual",
                data_name="mean",
                data=self.annual_data_mean,
                description=self.description,
                startdate=self.annual_data_mean.time.values[0],
                enddate=self.annual_data_mean.time.values[-1],
            )
            self.save_netcdf(
                var=self.var,
                freq="annual",
                data_name="std",
                data=self.annual_data_std,
                description=self.description,
                startdate=self.annual_data_std.time.values[0],
                enddate=self.annual_data_std.time.values[-1],
            )
        else:
            self.logger.info("No annual ensemble data is provided")
