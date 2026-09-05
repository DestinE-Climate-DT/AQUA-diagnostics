import gc
import sys

import xarray as xr

from .base import BaseMixin

xr.set_options(keep_attrs=True)

class VariabilityMap(BaseMixin):
    """
    Variability Map Computation

    Note: Variability means STD in this diagnostic
    """

    def __init__(
        self,
        diagnostic_name: str = "VariabilityMap",
        catalog: str = None,
        model: str = None,
        exp: str = None,
        source: str = None,
        startdate: str = None,
        enddate: str = None,
        freq: str = None,
        region: str = None,
        regrid: str = None,
        lon_limits: list[float] = None,
        lat_limits: list[float] = None,
        var: str = None,
        long_name: str = None,
        short_name: str = None,
        units: str = None,
        save_netcdf: bool = True,
        rebuild: bool = True,
        outputdir: str = "./",
        reader_kwargs: dict = {},
        fix: bool = True,
        loglevel: str = "WARNING",
    ):
        """
        Initialize the 'VariabilityMap' class.

        This class is designed to load an xarray.Dataset and computes STD.

        Args:
            diagnostic_name (str): Default is 'VariabilityMap'.
            catalog (str): catalog. It is Mandatory, if 'save_netcdf=True'.
            model (str): Name of the data
            exp (str): Name of the experiment
            source (str): the source. It is important to give these dates and input.
                Otherwise the whole dataset is retrieved.
            startdate (str): Start date.
            enddate  (str): End date.
            freq (str): Frequency of the data. TODO.
            region (str): For subregion selection. Default is 'None'.
                In case of sub-region STD computation, this variable is mandatory.
            regrid (str): Regrid option for the data. NOTE: the regridding will be applied before computing the STD.
            If 'lon_limits' and 'lat_limits' are None, they are taken from region file in AQUA.
            lon_limits (list[float]): list of lon limits. Default is 'None'.
            lat_limits (list[float]): list of lat limits. Default is 'None'.
            var (str): Variable name from data. Default is 'None'.
            long_name (str): If not given extracted from the data.
            short_name (str): If not given extracted from the data.
            units (str): If not given extracted from the data.

            save_netcdf (bool): Default is 'True'.
            rebuild (bool): Recomputes and saves the netcdf. Default is "True".
            outputdir (str): output directory. Default is './'
            loglevel (str): Default WARNING.

        Keyword Args:
            zoom (int, optional): HEALPix grid zoom level (e.g. zoom=10 is h1024). Allows for multiple gridname definitions.
            realization (int, optional): The ensemble realization number, included in the output filename.
            **kwargs: Additional arbitrary keyword arguments to be passed as additional parameters to the intake catalog entry

        """

        super().__init__(
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            startdate=startdate,
            enddate=enddate,
            region=region,
            regrid=regrid,
            lon_limits=lon_limits,
            lat_limits=lat_limits,
            reader_kwargs=reader_kwargs,
            var=var,
            long_name=long_name,
            short_name=short_name,
            units=units,
            outputdir=outputdir,
            rebuild=rebuild,
            fix=fix,
            loglevel=loglevel,
        )

        self.save_netcdf = save_netcdf
        self.freq = freq
        # To be assigned inside after STD computation run()
        self.data_std = None
        self.startdate = startdate
        self.enddate = enddate

    def run(self):
        """
        Args:
            create_catalog_entry (bool): Option for creating catalog entry. Default is 'False'.

        This function performs following three functions:
        a) Retrieve data and regrid if given then
        b) Compute STD
        c) Save netcdf
        """

        super().retrieve()
        if self.data is None and self.var is None:
            self.logger.warning(f"Variable {self.var} not found in the data. Check the variable name and the data source.")
        else:
            # Compute STD
            self.data_std = self.data.std(dim="time", skipna=True).compute()
            self.logger.info("Variability Map computation complete")
            # Removing the reference and releasing the memory from the Object reference, which is no longer needed
            del self.data
            gc.collect()

            # Remove the non-serializable attribute
            # in case of downloading data from Polytope this attribute was found
            if '_earthkit' in self.data_std.attrs:
                del self.data_std.attrs['_earthkit']
            # Save STD as netcdf
            if self.save_netcdf:
                self.logger.info(f"Output std netcdf file is saved at {self.outputdir}.")
                self.netcdf_save(data=self.data_std, create_catalog_entry=True)
            else:
                self.logger.info("Output in netcdf is not saved.")
