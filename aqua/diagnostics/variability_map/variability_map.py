import xarray as xr

from aqua.core.logger import log_configure
from aqua.diagnostics.base import Diagnostic

xr.set_options(keep_attrs=True)


class VariabilityMap(Diagnostic):
    """
    Variability Map Computation

    Note: Variability means STD in this diagnostic
    """

    # The STD along time needs at least two time steps to be defined
    MINIMUM_MONTHS_REQUIRED = 2

    def __init__(
        self,
        catalog: str = None,
        model: str = None,
        exp: str = None,
        source: str = None,
        startdate: str = None,
        enddate: str = None,
        region: str = None,
        regrid: str = None,
        var: str = None,
        long_name: str = None,
        short_name: str = None,
        units: str = None,
        reader_kwargs: dict = None,
        diagnostic_name: str = "variability_map",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the 'VariabilityMap' class.

        This class is designed to load an xarray.Dataset and computes STD.

        Args:
            catalog (str): catalog. If None, determined by the Reader.
            model (str): Name of the data
            exp (str): Name of the experiment
            source (str): the source.
            startdate (str): Start date. If None, all available data is used.
            enddate  (str): End date. If None, all available data is used.
            region (str): Region name from the regions file, used as metadata and in the filename. Default is 'None'.
            regrid (str): Regrid option for the data. NOTE: the regridding will be applied before computing the STD.
            var (str): Variable name from data. Default is 'None'.
            long_name (str): If not given extracted from the data.
            short_name (str): If not given extracted from the data.
            units (str): If given, the data is converted to these units.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader (e.g. fix, realization).
            diagnostic_name (str): Name used for the output files, usually the collection name.
                Default is 'variability_map'.
            loglevel (str): Default WARNING.
        """
        super().__init__(
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            startdate=startdate,
            enddate=enddate,
            regrid=regrid,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_level=loglevel, log_name="VariabilityMap")
        self.diagnostic_name = diagnostic_name
        self.region, self.lon_limits, self.lat_limits = self._set_region(region=region)

        self.var = var
        self.long_name = long_name
        self.short_name = short_name
        self.units = units
        self.reader_kwargs = reader_kwargs or {}

        # To be assigned after STD computation in run()
        self.data_std = None

    def retrieve(self):
        """
        Retrieve the data for the given variable as a DataArray, with units and names set.
        """
        super().retrieve(var=self.var, reader_kwargs=self.reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)
        self.data = self.data[self.var]

        if self.units is not None:
            self.data = self._check_data(data=self.data, var=self.var, units=self.units)
        else:
            self.units = self.data.attrs.get("units")
        # long_name is always defined for the plot title and description
        self.data.attrs["long_name"] = self.long_name or self.data.attrs.get("long_name", self.var)
        # short_name is used as the name of the variable in plots and filenames
        self.data.attrs["short_name"] = self.short_name or self.var
        self.data.name = self.data.attrs["short_name"]
        if self.region is not None:
            self.data.attrs["AQUA_region"] = self.region

    def run(self, outputdir: str = "./", rebuild: bool = True, save_netcdf: bool = True):
        """
        Retrieve the data (regridded if requested), compute the STD along time and optionally save it.

        Args:
            outputdir (str): output directory. Default is './'.
            rebuild (bool): If True, overwrite an existing netcdf. Default is True.
            save_netcdf (bool): If True, save the STD map as netcdf. Default is True.
        """
        self.retrieve()
        self.data_std = self.data.std(dim="time", skipna=True).compute()
        self.logger.info("Variability Map computation complete")
        # The time series is no longer needed, release it
        self.data = None

        if save_netcdf:
            self.save_netcdf(outputdir=outputdir, rebuild=rebuild)

    def save_netcdf(self, outputdir: str = "./", rebuild: bool = True):
        """
        Save self.data_std to a netcdf file.

        Args:
            outputdir (str): output directory. Default is './'.
            rebuild (bool): If True, overwrite an existing file. Default is True.
        """
        extra_keys = {
            "var": self.data_std.name,
            # A region key even in the global case, to have a stable filename
            "region": self.region.replace(" ", "").lower() if self.region is not None else "global",
        }
        self.logger.info("Saving the variability map to netcdf in %s", outputdir)
        super().save_netcdf(
            data=self.data_std,
            diagnostic=self.diagnostic_name,
            diagnostic_product="variability_map",
            outputdir=outputdir,
            rebuild=rebuild,
            extra_keys=extra_keys,
        )
