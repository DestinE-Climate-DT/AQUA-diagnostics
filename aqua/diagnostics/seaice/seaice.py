"""Sea ice diagnostic: regional extent and volume series, fraction and thickness climatologies.

Example:
    seaice = SeaIce(model="IFS-NEMO", exp="historical-1990", source="lra-r100-monthly", regions=["arctic", "antarctic"])
    seaice.run(method="extent")
    seaice.monthly  # xr.Dataset with one variable per region: sea_ice_extent_arctic, sea_ice_extent_antarctic
"""

import xarray as xr

from aqua.core.logger import log_configure, log_history
from aqua.core.util import DEFAULT_REALIZATION, merge_attrs, time_to_string, to_list
from aqua.diagnostics.base import Diagnostic

xr.set_options(keep_attrs=True)

# method -> default variable and units of the result
METHODS = {
    "extent": {"var": "siconc", "units": "million km^2"},
    "volume": {"var": "sithick", "units": "thousands km^3"},
    "fraction": {"var": "siconc", "units": "1"},
    "thickness": {"var": "sithick", "units": "m"},
}
# methods integrated over each region; the others are monthly climatology maps
INTEGRATED = ("extent", "volume")


class SeaIce(Diagnostic):
    """
    Sea ice metrics of one dataset over a list of regions.

    ``run()`` computes one method at a time. Extent (million km^2) and volume (thousands km^3) are integrated over
    each region: as a time series, optionally with its annual means, or as a seasonal cycle, both optionally with the
    standard deviation needed to draw an uncertainty band. Fraction (1) and thickness (m) are monthly climatology maps.
    Every result is a Dataset with one variable per region, named ``sea_ice_<method>_<region>``.

    Args:
        model (str): The model name.
        exp (str): The experiment name.
        source (str): The data source.
        catalog (str, optional): The catalog name. If None, it is determined by the Reader.
        regrid (str, optional): The target grid for regridding. If None, no regridding is done.
        startdate (str, optional): The start date of the analysis. If None, all data are used.
        enddate (str, optional): The end date of the analysis. If None, all data are used.
        std_startdate (str, optional): The start date of the standard deviation period.
        std_enddate (str, optional): The end date of the standard deviation period.
            If both are None, the standard deviation is computed over the analysis period.
        regions (str or list, optional): Region names of the regions file. Default is ['arctic', 'antarctic'].
        threshold (float, optional): Sea ice concentration above which a cell counts for the extent. Default is 0.15.
        regions_file (str, optional): Path of a custom regions file. If None, the centralized one is used.
        diagnostic_name (str, optional): The name used for the output files. Default is 'seaice'.
        loglevel (str, optional): The log level. Default is 'WARNING'.
    """

    # One month already gives a point of the series or a map: run() asks for 12 for the seasonal cycle
    MINIMUM_MONTHS_REQUIRED = 1

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        std_startdate: str = None,
        std_enddate: str = None,
        regions: list = None,
        threshold: float = 0.15,
        regions_file: str = None,
        diagnostic_name: str = "seaice",
        loglevel: str = "WARNING",
    ):
        super().__init__(
            model=model,
            exp=exp,
            source=source,
            regrid=regrid,
            catalog=catalog,
            startdate=startdate,
            enddate=enddate,
            std_startdate=std_startdate,
            std_enddate=std_enddate,
            loglevel=loglevel,
        )
        self.logger = log_configure(loglevel, "SeaIce")
        self.diagnostic_name = diagnostic_name
        self.threshold = threshold
        if (std_startdate is None) != (std_enddate is None):
            raise ValueError("std_startdate and std_enddate must be given together")
        # retrieve() replaces startdate/enddate with the available ones: the requested ones identify the files
        self._requested_dates = (startdate, enddate)

        self.regions_file = regions_file
        self.regions_definition = self._load_regions_from_file(regions_file_path=regions_file)
        self.regions = to_list(regions)
        if not self.regions:
            self.regions = ["arctic", "antarctic"]
            self.logger.warning("No regions specified. Falling back to default sea-ice regions: %s", self.regions)
        invalid = [region for region in self.regions if region not in self.regions_definition]
        if invalid:
            raise ValueError(
                f"Invalid region name(s): {invalid}. Please check regions names are lower case "
                f"or the region file at: '{regions_file or 'centralized regions file'}'."
            )

        self.method = None
        self.var = None
        self._reset_results()

    def _reset_results(self):
        """Forget the results of a previous run."""
        self.monthly = None
        self.annual = None
        self.seasonalcycle = None
        self.climatology = None
        self.std_monthly = None
        self.std_annual = None

    def _set_method(self, method: str, var: str = None):
        """Check the method and set it with its variable."""
        if method not in METHODS:
            raise ValueError(f"Invalid method '{method}'. Please choose from: {list(METHODS)}")
        self.method = method
        self.var = var or METHODS[method]["var"]

    def run(
        self,
        method: str = "extent",
        var: str = None,
        seasonal_cycle: bool = False,
        annual: bool = False,
        std: bool = False,
        reader_kwargs: dict = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_netcdf: bool = True,
    ):
        """
        Retrieve the data, compute the sea ice metric and optionally save it.

        Extent and volume fill ``monthly`` (and ``annual``), or ``seasonalcycle`` if requested, plus ``std_monthly``
        (and ``std_annual``) with ``std``. Fraction and thickness fill ``climatology`` and ignore the other options.

        Args:
            method (str): One of 'extent', 'volume', 'fraction', 'thickness'. Default is 'extent'.
            var (str, optional): The variable to use. Default is 'siconc' for extent and fraction,
                'sithick' for volume and thickness.
            seasonal_cycle (bool, optional): Compute the seasonal cycle instead of the time series. Default is False.
            annual (bool, optional): Also compute the annual means of the time series, on complete years only.
                Default is False.
            std (bool, optional): Also compute the standard deviation, by calendar month (and of the annual means
                with ``annual``). Default is False.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_netcdf (bool, optional): If True, save the results to NetCDF. Default is True.

        Raises:
            ValueError: If the method is not valid.
            NotEnoughDataError: If less than 12 months are available for the seasonal cycle.
        """
        self._set_method(method, var)
        self._reset_results()
        months_required = 12 if seasonal_cycle else self.MINIMUM_MONTHS_REQUIRED
        self.retrieve(var=self.var, reader_kwargs=reader_kwargs or {}, months_required=months_required)

        if self.method not in INTEGRATED:
            self.compute_climatology()
        else:
            self.compute_timeseries(annual=annual and not seasonal_cycle)
            if std:
                self.compute_std(annual=annual and not seasonal_cycle)
            if seasonal_cycle:
                self.compute_seasonalcycle()
        if save_netcdf:
            self.save_netcdf(outputdir=outputdir, rebuild=rebuild)

    def compute_timeseries(self, annual: bool = False):
        """
        Integrate the sea ice over each region and store the series in self.monthly (and self.annual).

        Args:
            annual (bool, optional): Also compute the annual means of complete years. Default is False.
        """
        # shortcut: the series keeps the frequency of the source, which is monthly;
        # resample here with reader.timmean when daily sources are supported
        series = self._integrate(self.data)
        self.monthly = self._dataset(series)
        if annual:
            self.annual = self._dataset(self._annual_means(series))

    def compute_seasonalcycle(self):
        """Average self.monthly by calendar month and store the cycle in self.seasonalcycle."""
        cycle = [da.groupby("time.month").mean("time") for da in self.monthly.data_vars.values()]
        self.seasonalcycle = self._dataset([log_history(da, "Seasonal cycle: mean by calendar month") for da in cycle])

    def compute_std(self, annual: bool = False):
        """
        Compute the standard deviation of the integrated series and store it in self.std_monthly (and self.std_annual).

        It covers the std period if std dates were given, the analysis period otherwise.

        Args:
            annual (bool, optional): Also compute the standard deviation of the annual means. Default is False.
        """
        if self.std_data is not None:
            series = self._integrate(self.std_data)
        else:
            series = list(self.monthly.data_vars.values())
        self.std_monthly = self._dataset([da.groupby("time.month").std("time") for da in series], std=True)
        if annual:
            yearly = self._annual_means(series)
            if yearly is not None and yearly[0].time.size < 2:
                self.logger.warning("Only one complete year: the standard deviation of the annual means is zero")
            self.std_annual = self._dataset([da.std("time") for da in yearly] if yearly else None, std=True)

    def compute_climatology(self):
        """Compute the monthly climatology map over each region and store it in self.climatology."""
        data = self.data[self.var]
        maps = [
            self.select_region(data, region=region, regions_file_path=self.regions_file)["data"]
            .groupby("time.month")
            .mean("time")
            for region in self.regions
        ]
        self.climatology = self._dataset(maps)

    def _integrate(self, data: xr.Dataset):
        """Mask the sea ice and integrate it over each region: one series in memory per region of self.regions."""
        var = data[self.var]
        if self.method == "extent":
            masked = var.where((var > self.threshold) & (var <= 1.0))  # above 1 only fill values are found
        else:
            masked = var.where((var > 0) & (var < 99.0))

        series = []
        for region in self.regions:
            self.logger.info("Computing sea ice %s for %s", self.method, region)
            # Select before notnull(): a selection inside fldarea would count the cells outside the region.
            # to_180=False keeps the longitudes aligned with the cell areas of the Reader.
            selected = self.select_region(
                masked, region=region, regions_file_path=self.regions_file, drop=False, to_180=False
            )["data"]
            if self.method == "extent":
                integral = self.reader.fldarea(selected.notnull()) / 1e12  # m^2 -> million km^2
            else:
                integral = self.reader.fldintg(selected) / 1e12  # m^3 -> thousands km^3
            # fldarea returns the attributes of the cell areas: bring back the provenance of the data
            merge_attrs(integral.attrs, selected.attrs)
            series.append(log_history(integral, f"Sea ice {self.method} integrated over {region}").rename(region))
        # one computation for all the regions: the series are small and reused by the annual means and the std
        return list(xr.merge(series).load().data_vars.values())

    def _annual_means(self, series: list):
        """Annual means of the complete years of each series, None if there is no complete year."""
        # with a single time step the Reader skips the completeness check
        if series[0].time.size > 1:
            yearly = [self.reader.timmean(da, freq="annual", exclude_incomplete=True, center_time=True) for da in series]
            if yearly[0].time.size > 0:
                return yearly
        self.logger.warning("No complete year of data: annual means are not computed")
        return None

    def _dataset(self, arrays: list, std: bool = False):
        """Name and describe the result of each region, then merge them into one Dataset loaded in memory."""
        if arrays is None:
            return None
        if std and self.std_data is not None:
            dates = {"AQUA_std_startdate": self.std_startdate, "AQUA_std_enddate": self.std_enddate}
        elif std:
            dates = {"AQUA_std_startdate": self.startdate, "AQUA_std_enddate": self.enddate}
        else:
            dates = {"AQUA_startdate": self.startdate, "AQUA_enddate": self.enddate}
        prefix = "std_" if std else ""

        for region, da in zip(self.regions, arrays):
            da.name = f"{prefix}sea_ice_{self.method}_{region}"
            for key in ("AQUA_startdate", "AQUA_enddate"):
                da.attrs.pop(key, None)
            da.attrs.update({key: time_to_string(date) for key, date in dates.items()})
            da.attrs["units"] = METHODS[self.method]["units"]
            da.attrs["long_name"] = (
                f"{'Std ' if std else ''}Sea ice {self.method} "
                f"{'integrated ' if self.method in INTEGRATED else ''}over {da.attrs['AQUA_region']}"
            )
            da.attrs["standard_name"] = f"{region}_{prefix}sea_ice_{self.method}"
            da.attrs["AQUA_method"] = self.method
            if "time" in da.coords:
                da["time"].attrs["long_name"] = "Time"
                da["time"].attrs.setdefault("standard_name", "time")

        # the maps of different regions do not share the latitudes: keep them all
        dataset = xr.merge(arrays, join="outer", combine_attrs="drop_conflicts")
        dataset.attrs.update({"AQUA_catalog": self.catalog, "AQUA_model": self.model, "AQUA_exp": self.exp})
        # the history of each region differs and is dropped by the merge: the dataset has its own
        dataset = log_history(dataset, f"Sea ice {self.method} computed by the SeaIce diagnostic")
        # load() looks for the requested period, which differs from the available one when retrieve() clipped it
        for key, date in zip(("AQUA_startdate", "AQUA_enddate"), self._requested_dates):
            if date is not None:
                dataset.attrs[key] = time_to_string(date)
        return dataset.load()

    def _products(self, seasonal_cycle: bool) -> dict:
        """NetCDF product of each result attribute, for the kind of run."""
        if self.method not in INTEGRATED:
            return {"climatology": "bias"}
        if seasonal_cycle:
            return {"seasonalcycle": "seasonalcycle", "std_monthly": "seasonalcycle_std"}
        return {
            "monthly": "timeseries",
            "annual": "timeseries_annual",
            "std_monthly": "timeseries_std",
            "std_annual": "timeseries_annual_std",
        }

    def _extra_keys(self) -> dict:
        """Filename keys of the results, shared by save_netcdf and load so that both address the same files."""
        return {"method": self.method, "source": self.source, "regions_domain": "_".join(self.regions)}

    def save_netcdf(self, outputdir: str = ".", rebuild: bool = True):
        """
        Save the results of the last run to NetCDF, one file per result.

        Args:
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
        """
        for attribute, product in self._products(self.seasonalcycle is not None).items():
            data = getattr(self, attribute)
            if data is not None:
                super().save_netcdf(
                    data=data,
                    diagnostic=self.diagnostic_name,
                    diagnostic_product=product,
                    outputdir=outputdir,
                    rebuild=rebuild,
                    extra_keys=self._extra_keys(),
                )

    def load(
        self,
        method: str = "extent",
        seasonal_cycle: bool = False,
        outputdir: str = ".",
        reader_kwargs: dict = None,
    ):
        """
        Populate the results from the NetCDF files written by a previous run.

        A missing file, or one computed over a different period, leaves its result untouched.

        Args:
            method (str): The method that was computed. Default is 'extent'.
            seasonal_cycle (bool, optional): True if the run computed the seasonal cycle. Default is False.
            outputdir (str, optional): The directory where the data was saved. Default is '.'.
            reader_kwargs (dict, optional): The Reader keyword arguments of the run, to match its realization.
        """
        self._set_method(method)
        self.realization = (reader_kwargs or {}).get("realization", DEFAULT_REALIZATION)
        for attribute, product in self._products(seasonal_cycle).items():
            # shortcut: only the analysis period is checked, not the std one;
            # add it when the CLI plots from files
            data = self.load_netcdf(
                diagnostic=self.diagnostic_name,
                diagnostic_product=product,
                outputdir=outputdir,
                extra_keys=self._extra_keys(),
                startdate=self._requested_dates[0],
                enddate=self._requested_dates[1],
            )
            if data is not None:
                setattr(self, attribute, data)
