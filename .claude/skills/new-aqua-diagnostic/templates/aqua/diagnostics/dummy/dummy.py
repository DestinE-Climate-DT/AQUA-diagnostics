"""Dummy tool: annual means of a field averaged over a region."""

import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION
from aqua.diagnostics.base import Diagnostic

xr.set_options(keep_attrs=True)


class Dummy(Diagnostic):
    """Compute the annual mean of a variable averaged over a region, for one dataset."""

    # One full year is needed to have at least one complete annual mean
    MINIMUM_MONTHS_REQUIRED = 12
    # Name of the NetCDF product; save_netcdf() and load() must agree on it
    DIAGNOSTIC_PRODUCT = "annual_mean"

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        region: str = None,
        diagnostic_name: str = "dummy",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the Dummy class.

        Args:
            model (str): The model to be used.
            exp (str): The experiment to be used.
            source (str): The source to be used.
            catalog (str, optional): The catalog to be used. If None, it is determined by the Reader.
            regrid (str, optional): The target grid for regridding. If None, no regridding is done.
            startdate (str, optional): The start date of the analysis. If None, all data are used.
            enddate (str, optional): The end date of the analysis. If None, all data are used.
            region (str, optional): A region name of the regions file. If None, the global domain is used.
            diagnostic_name (str, optional): The name used for the output files,
                usually the collection name. Default is 'dummy'.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        super().__init__(
            model=model,
            exp=exp,
            source=source,
            catalog=catalog,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_level=loglevel, log_name="Dummy")
        self.diagnostic_name = diagnostic_name
        self.region, self.lon_limits, self.lat_limits = self._set_region(region=region)
        self.result = None

    def run(
        self,
        var: str,
        reader_kwargs: dict = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_netcdf: bool = True,
    ):
        """
        Retrieve the data, compute the annual mean and optionally save it.

        Args:
            var (str): The variable to analyse.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_netcdf (bool, optional): If True, save the result to NetCDF. Default is True.

        Raises:
            NotEnoughDataError: If less than MINIMUM_MONTHS_REQUIRED months are available.
        """
        self.retrieve(var=var, reader_kwargs=reader_kwargs or {}, months_required=self.MINIMUM_MONTHS_REQUIRED)
        self.compute_result(var=var)
        if save_netcdf:
            self.save_netcdf(outputdir=outputdir, rebuild=rebuild)

    def compute_result(self, var: str):
        """
        Compute the regional annual mean and store it in self.result.

        Args:
            var (str): The variable to analyse.
        """
        data = self.reader.fldmean(self.data[var], lon_limits=self.lon_limits, lat_limits=self.lat_limits)
        result = self.reader.timmean(data, freq="annual", exclude_incomplete=True)
        # Everything the plot will need travels with the result as AQUA_* attributes
        result.attrs.update({k: v for k, v in self.data.attrs.items() if k.startswith("AQUA_")})
        if self.region is not None:
            result.attrs["AQUA_region"] = self.region
        result.name = var
        self.result = result.load()

    def _extra_keys(self, var: str) -> dict:
        """
        Build the filename keys of a result.

        Shared by save_netcdf and load, so that both address the same file.

        Args:
            var (str): The variable name appearing in the filename.

        Returns:
            dict: The extra keys, in the order in which they appear in the filename.
        """
        keys = {"var": var}
        if self.region is not None:
            keys["region"] = self.region
        return keys

    def save_netcdf(self, outputdir: str = ".", rebuild: bool = True):
        """
        Save self.result to a NetCDF file.

        Args:
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite an existing file. Default is True.
        """
        # load_netcdf() matches startdate/enddate against the dataset-level attributes, while a bare DataArray
        # keeps all of them on the variable: store a Dataset with the attributes at both levels, otherwise a
        # plot-only run with dates in the configuration silently finds nothing to load.
        data = self.result.to_dataset()
        data.attrs.update(self.result.attrs)
        super().save_netcdf(
            data=data,
            diagnostic=self.diagnostic_name,
            diagnostic_product=self.DIAGNOSTIC_PRODUCT,
            outputdir=outputdir,
            rebuild=rebuild,
            extra_keys=self._extra_keys(self.result.name),
        )

    def load(self, var: str, outputdir: str = ".", reader_kwargs: dict = None):
        """
        Populate self.result from the NetCDF file written by a previous run.

        A missing file, or one computed over a different period, leaves self.result untouched.

        Args:
            var (str): The variable that was analysed.
            outputdir (str, optional): The directory where the data was saved. Default is '.'.
            reader_kwargs (dict, optional): The Reader keyword arguments of the run, to match its realization.
        """
        self.realization = (reader_kwargs or {}).get("realization", DEFAULT_REALIZATION)
        data = self.load_netcdf(
            diagnostic=self.diagnostic_name,
            diagnostic_product=self.DIAGNOSTIC_PRODUCT,
            outputdir=outputdir,
            extra_keys=self._extra_keys(var),
            as_dataarray=True,
            startdate=self.startdate,
            enddate=self.enddate,
        )
        if data is not None:
            self.result = data
