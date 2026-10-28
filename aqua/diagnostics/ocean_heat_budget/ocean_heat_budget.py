"""OceanHeatBudget tool: global net surface heat flux against the tendency of the ocean heat content."""

import numpy as np
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION
from aqua.diagnostics.base import Diagnostic

xr.set_options(keep_attrs=True)


class OceanHeatBudget(Diagnostic):
    """
    Compare, over the global ocean, the net surface heat flux with the time derivative of the heat content.

    If the surface fluxes are the only source of heat of the ocean layer considered, the two curves coincide:
    their difference measures how much of the budget is closed (advection through the layer boundaries,
    drift, sea ice and numerical issues make up the rest). One dataset is analysed: it must contain
    both the atmospheric fluxes and the ocean variables, as the LRA datasets do.
    """

    # One full year is needed to resolve the annual cycle of the budget
    MINIMUM_MONTHS_REQUIRED = 12
    # Name of the NetCDF product; save_netcdf() and load() must agree on it
    DIAGNOSTIC_PRODUCT = "budget"
    # Surface net fluxes (W m**-2, positive downward, i.e. into the ocean)
    FLUX_VARS = ["slhtf", "snlwrf", "snswrf", "ishf"]
    # Snowfall rate water equivalent (kg m**-2 s**-1): the ocean must melt it, which removes heat
    SNOW_VAR = "tsrwe"
    # Sea surface temperature: its missing values define the ocean
    MASK_VAR = "tos"
    # Latent heat of fusion of water (J kg**-1)
    LATENT_HEAT_FUSION = 3.34e5

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        diagnostic_name: str = "ocean_heat_budget",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the OceanHeatBudget class.

        Args:
            model (str): The model to be used.
            exp (str): The experiment to be used.
            source (str): The source to be used.
            catalog (str, optional): The catalog to be used. If None, it is determined by the Reader.
            regrid (str, optional): The target grid for regridding. If None, no regridding is done.
            startdate (str, optional): The start date of the analysis. If None, all data are used.
            enddate (str, optional): The end date of the analysis. If None, all data are used.
            diagnostic_name (str, optional): The name used for the output files,
                usually the collection name. Default is 'ocean_heat_budget'.
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
        self.logger = log_configure(log_level=loglevel, log_name="OceanHeatBudget")
        self.diagnostic_name = diagnostic_name
        self.result = None

    def run(
        self,
        hc_var: str = "hc700m",
        reader_kwargs: dict = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_netcdf: bool = True,
    ):
        """
        Retrieve the data, compute the budget terms and optionally save them.

        Args:
            hc_var (str, optional): The ocean heat content variable (J m**-2). Default is 'hc700m'.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_netcdf (bool, optional): If True, save the result to NetCDF. Default is True.

        Raises:
            NotEnoughDataError: If less than MINIMUM_MONTHS_REQUIRED months are available.
        """
        variables = [*self.FLUX_VARS, self.SNOW_VAR, self.MASK_VAR, hc_var]
        self.retrieve(var=variables, reader_kwargs=reader_kwargs or {}, months_required=self.MINIMUM_MONTHS_REQUIRED)
        self.compute_result(hc_var=hc_var)
        if save_netcdf:
            self.save_netcdf(hc_var=hc_var, outputdir=outputdir, rebuild=rebuild)

    def compute_result(self, hc_var: str = "hc700m"):
        """
        Compute the global mean net surface flux and heat content tendency and store them in self.result.

        Args:
            hc_var (str, optional): The ocean heat content variable (J m**-2). Default is 'hc700m'.
        """
        data = self.data
        ocean = data[self.MASK_VAR].isel(time=0).notnull()

        net_flux = sum(data[v] for v in self.FLUX_VARS) - data[self.SNOW_VAR] * self.LATENT_HEAT_FUSION
        # Land is masked with NaN, not zero: the global mean must be over the ocean, like the heat content one
        net_flux = self.reader.fldmean(net_flux.where(ocean))
        heat_content = self.reader.fldmean(data[hc_var])

        # Forward difference between consecutive steps, labelled at the first one; the last step has none
        seconds = heat_content.time.diff("time", label="lower") / np.timedelta64(1, "s")
        tendency = heat_content.diff("time", label="lower") / seconds

        hc_name = data[hc_var].attrs.get("long_name", hc_var)
        net_flux.attrs.update(short_name="net_surface_flux", long_name="Net surface heat flux", units="W m**-2")
        tendency.attrs.update(short_name=f"{hc_var}_tendency", long_name=f"Time derivative of {hc_name}", units="W m**-2")

        result = xr.Dataset({"net_surface_flux": net_flux.sel(time=tendency.time), "hc_tendency": tendency})
        # Everything the plot will need travels with the result as AQUA_* attributes
        # (retrieve puts the period on the dataset and the Reader provenance on the variables)
        provenance = {**data[hc_var].attrs, **data.attrs}
        result.attrs.update({k: v for k, v in provenance.items() if k.startswith("AQUA_")})
        self.result = result.load()

    def _extra_keys(self, hc_var: str) -> dict:
        """
        Build the filename keys of a result.

        Shared by save_netcdf and load, so that both address the same file.

        Args:
            hc_var (str): The heat content variable appearing in the filename.

        Returns:
            dict: The extra keys, in the order in which they appear in the filename.
        """
        return {"var": hc_var}

    def save_netcdf(self, hc_var: str = "hc700m", outputdir: str = ".", rebuild: bool = True):
        """
        Save self.result to a NetCDF file.

        Args:
            hc_var (str, optional): The heat content variable of the run. Default is 'hc700m'.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite an existing file. Default is True.
        """
        # load_netcdf() matches startdate/enddate against the dataset-level attributes: self.result is a Dataset
        # with the AQUA_* attributes there, so a plot-only run with dates in the configuration finds the file.
        super().save_netcdf(
            data=self.result,
            diagnostic=self.diagnostic_name,
            diagnostic_product=self.DIAGNOSTIC_PRODUCT,
            outputdir=outputdir,
            rebuild=rebuild,
            extra_keys=self._extra_keys(hc_var),
        )

    def load(self, hc_var: str = "hc700m", outputdir: str = ".", reader_kwargs: dict = None):
        """
        Populate self.result from the NetCDF file written by a previous run.

        A missing file, or one computed over a different period, leaves self.result untouched.

        Args:
            hc_var (str, optional): The heat content variable of the run. Default is 'hc700m'.
            outputdir (str, optional): The directory where the data was saved. Default is '.'.
            reader_kwargs (dict, optional): The Reader keyword arguments of the run, to match its realization.
        """
        self.realization = (reader_kwargs or {}).get("realization", DEFAULT_REALIZATION)
        data = self.load_netcdf(
            diagnostic=self.diagnostic_name,
            diagnostic_product=self.DIAGNOSTIC_PRODUCT,
            outputdir=outputdir,
            extra_keys=self._extra_keys(hc_var),
            as_dataarray=False,
            startdate=self.startdate,
            enddate=self.enddate,
        )
        if data is not None:
            self.result = data
