"""Module for computing trends of one or more variables along the time dimension."""

import pandas as pd
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.reader import Trender
from aqua.core.util import to_list
from aqua.diagnostics.base import Diagnostic

xr.set_options(keep_attrs=True)


class Trends(Diagnostic):
    """
    Class to compute linear trends along the time dimension for one or more variables.
    The trend is computed via polynomial fit and rescaled to per-year units based on the inferred time frequency of the data.
    Supported 2D and 3d fields.
    """

    MINIMUM_MONTHS_REQUIRED = 12

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        diagnostic_name: str = "trends",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the Trends class.
        There is no region selection in the initialization. At run time, if a list of regions is provided,
        the trend will be computed first globally and then a loop will select and store the trend for each region.
        If no region or only one region is provided, the trend will be computed only once and stored in a single file.

        If a 3D data is provided, the trend will be computed for each vertical level.
        If a 2D data is provided, the trend will be computed for the single level.

        Args:
            model (str): Model name.
            exp (str): Experiment name.
            source (str): Data source.
            catalog (str, optional): Catalog name. Resolved by the Reader if None.
            regrid (str, optional): Target grid for regridding. No regridding if None.
            startdate (str, optional): Analysis start date.
            enddate (str, optional): Analysis end date.
            diagnostic_name (str, optional): Diagnostic name used in output filenames.
                Defaults to ``'trends'``.
            loglevel (str, optional): Logging level. Defaults to ``'WARNING'``.
        """
        super().__init__(
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            regrid=regrid,
            startdate=startdate,
            enddate=enddate,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_level=loglevel, log_name="Trends")
        self.diagnostic_name = diagnostic_name

        # Store the computed trend
        # The structure of the trend data will be a dictionary with region names as keys and the corresponding
        # trend datasets as values. If no region is specified, the key will be 'global'.
        # If multiple variables are provided, they will be stored in the same dataset.
        self.trend_coef = {}

    def retrieve(self, var, reader_kwargs: dict = {}):
        """
        Retrieve the data for one or more variables.

        Args:
            var (str or list): Variable name(s) to retrieve.
            reader_kwargs (dict, optional): Extra keyword arguments forwarded to the Reader.
        """
        var = to_list(var)
        self.logger.info("Retrieving variable(s): %s", var)
        super().retrieve(var=var, reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)

    def run(
        self,
        var,
        dim_mean=None,
        region: str | list = None,
        lon_limits: list = None,
        lat_limits: list = None,
        regions_file_path: str = None,
        outputdir: str = "./",
        rebuild: bool = True,
        reader_kwargs: dict = {},
    ):
        """
        Run the full trend analysis workflow.
        We try to minimize the number of times the trend is computed.
        However, if a dim_mean is provided, the mean has to happen before the trend is computed,
        so the trend will be computed for each region separately.

        Steps:
            With only one region, no region or dim_mean: (trend_loop)
            retrieve → region selection → optional dimensional mean → trend → save.
            With a list of regions and no dim_mean: (region_loop)
            retrieve → trend → loop over regions: region selection → save.

        Args:
            var (str or list): Variable(s) to analyse.
            dim_mean (str or list, optional): Dimension(s) over which to take an areal mean
                before the trend is computed (e.g. ``['lat', 'lon']`` for a regional time series).
            region (str, optional): Region name in the centralized regions file.
            lon_limits (list, optional): Custom longitude limits ``[lon_min, lon_max]``.
            lat_limits (list, optional): Custom latitude limits ``[lat_min, lat_max]``.
            regions_file_path (str, optional): Custom regions YAML. Defaults to the
                centralized AQUA regions file.
            outputdir (str, optional): Output directory. Defaults to ``'./'``.
            rebuild (bool, optional): Whether to overwrite existing output files. Defaults to True.
            reader_kwargs (dict, optional): Extra keyword arguments forwarded to the Reader.
        """
        self.logger.info("Starting trend analysis")
        self.retrieve(var=var, reader_kwargs=reader_kwargs)

        # We have two different workflows depending on whether we have a list of regions and
        # no dim_mean or not. They are defined by the kind variable.
        kind = "region_loop" if isinstance(region, list) and dim_mean is None else "trend_loop"
        self.logger.debug("Trend analysis workflow kind: %s", kind)

        if kind == "trend_loop":
            # 1-2. select the region and optionally compute the mean over the specified dimensions
            if region is None and (lat_limits is None and lon_limits is None):
                self.logger.info("No region or custom limits provided, using global data")
                region = "global"
            else:
                region = region if region is not None else "custom limits"

            for reg in to_list(region):
                self.logger.info("Processing region: %s", reg)
                region, lon_limits, lat_limits = self._set_region(
                    region=reg if reg != "global" else None,
                    regions_file_path=regions_file_path,
                    lon_limits=lon_limits if reg != "global" else None,
                    lat_limits=lat_limits if reg != "global" else None,
                )
                data = self._apply_region(
                    self.data, region=region, lon_limits=lon_limits, lat_limits=lat_limits, dim_mean=dim_mean
                )

                self.logger.info("Computing trend coefficients")
                trend_coef = self.compute_trend(data=data, region=region)
                region_key = region if region is not None else "global"
                self.trend_coef[region_key] = trend_coef

        elif kind == "region_loop":
            self.logger.info("Computing trend coefficients for global data")
            # This first evaluation is already loading in memory the trend data for all
            # variables and all regions.
            trend_coef = self.compute_trend(data=self.data)
            self.trend_coef["global"] = trend_coef

            for reg in region:
                self.logger.info("Processing region: %s", reg)
                region, lon_limits, lat_limits = self._set_region(
                    region=reg,
                    regions_file_path=regions_file_path,
                    lon_limits=lon_limits,
                    lat_limits=lat_limits,
                )
                trend_data = self._apply_region(
                    trend_coef, region=region, lon_limits=lon_limits, lat_limits=lat_limits, dim_mean=None
                )
                self.trend_coef[region] = trend_data

        # We save all the regions at once.
        self.logger.info("Saving results to NetCDF for region: %s", region)
        self.save_netcdf(outputdir=outputdir, rebuild=rebuild)
        self.logger.info("Trend analysis completed for all regions")

    def _apply_region(self, data, region: str = None, lon_limits: list = None, lat_limits: list = None, dim_mean=None):
        """
        Apply region selection and optional field mean to a dataset.

        Args:
            data (xr.Dataset): Input data.
            region (str, optional): Region name.
            lon_limits (list, optional): Custom longitude limits ``[lon_min, lon_max]``.
            lat_limits (list, optional): Custom latitude limits ``[lat_min, lat_max]``.
            dim_mean (str or list, optional): Dimension(s) over which to compute the mean.

        Returns:
            xr.Dataset: The (possibly subset and averaged) data.
        """
        has_limits = lon_limits is not None or lat_limits is not None
        if region is not None or has_limits:
            label = region if region is not None else "custom limits"
            self.logger.info("Applying region selection: %s", label)
            if dim_mean is None:
                self.logger.debug("No dimension mean specified, selecting area only")
                data = self.reader.select_area(data=data, lat=lat_limits, lon=lon_limits, drop=True)
            data.attrs["AQUA_region"] = label

        # If dim_mean is specified we always need the fldmean to be applied, even if a region is selected.
        # The mean will be computed over the specified dimensions together with the lat/lon limits if provided.
        # The region name will be stored in the attributes.
        if dim_mean is not None:
            self.logger.debug("Averaging data over dimension(s): %s", dim_mean)
            data = self.reader.fldmean(data, dims=to_list(dim_mean), lat=lat_limits, lon=lon_limits)
            data.attrs["AQUA_dim_mean"] = dim_mean

        return data

    def compute_trend(self, data: xr.Dataset, region: str = None) -> xr.Dataset:
        """
        Compute the linear trend coefficients along ``time`` and rescale them to per-year.

        Args:
            data (xr.Dataset): Input dataset with a ``time`` dimension.
            region (str, optional): Region name to include in the output attributes.

        Returns:
            xr.Dataset: Trend coefficients (one per variable) with adjusted units.
        """
        self.logger.info("Calculating linear trend")
        trender = Trender(loglevel=self.loglevel)
        trend_data = trender.coeffs(data, dim="time", skipna=True, normalize=True)
        trend_data = trend_data.sel(degree=1)
        trend_data.attrs = data.attrs

        # HACK: polyfit drops non-time-indexed coordinates (e.g. lat/lon on ncells), restore them.
        # This is needed for Healpix and other non-standard grids where lat/lon coordinates depending on other dimensions.
        dropped_coords = {
            name: coord for name, coord in data.coords.items() if name not in trend_data.coords and "time" not in coord.dims
        }
        if dropped_coords:
            self.logger.debug("Restoring coordinates dropped by polyfit: %s", list(dropped_coords))
            trend_data = trend_data.assign_coords(dropped_coords)

        trend_dict = {}
        for var in data.data_vars:
            self.logger.debug("Adjusting trend for variable: %s", var)
            trend_data[var].attrs = data[var].attrs
            trend_dict[var] = self.adjust_trend_for_time_frequency(trend_data[var], data)
        trend_data = xr.Dataset(trend_dict)
        trend_data.attrs.update(data.attrs)
        if region is not None:
            trend_data.attrs["AQUA_region"] = region

        self.logger.debug("Loading trend data in memory")
        trend_data.load()
        return trend_data

    def adjust_trend_for_time_frequency(self, trend: xr.DataArray, y_array: xr.Dataset):
        """
        Scale the trend coefficient to per-year units based on the inferred input frequency.

        Args:
            trend (xr.DataArray): Trend coefficient (slope of the linear fit).
            y_array (xr.Dataset or xr.DataArray): Original data carrying the time coordinate.

        Returns:
            xr.DataArray: Trend scaled to per-year and with updated ``units`` attribute.
        """
        self.logger.debug("Adjusting trend for time frequency")
        time_frequency = y_array["time"].to_index().inferred_freq

        if time_frequency is None:
            self.logger.debug("Time frequency not inferred, checking for monthly data")
            time_index = pd.to_datetime(y_array["time"].values)
            time_diffs = time_index[1:] - time_index[:-1]
            is_monthly = all(time_diff.days >= 28 for time_diff in time_diffs)
            if is_monthly:
                time_frequency = "MS"
                self.logger.debug("Data inferred as monthly")
            else:
                self.logger.error("Unable to determine time frequency")
                raise ValueError("The frequency of the data must be in Daily/Monthly/Yearly")

        if time_frequency == "MS":
            self.logger.debug("Monthly data detected, scaling trend by 12")
            trend = trend * 12
        elif time_frequency == "H":
            self.logger.debug("Hourly data detected, scaling trend by 24*30*12")
            trend = trend * 24 * 30 * 12
        elif time_frequency in ("Y", "YE-DEC"):
            self.logger.debug("Yearly data detected, no scaling applied")
        else:
            self.logger.error("Unsupported time frequency: %s", time_frequency)
            raise ValueError(f"The frequency: {time_frequency} of the data must be in Daily/Monthly/Yearly")

        units = trend.attrs.get("units", "")
        trend.attrs["units"] = f"{units}/year" if units else "per year"
        self.logger.debug("Trend units updated to: %s", trend.attrs["units"])
        return trend

    def save_netcdf(
        self,
        diagnostic_product: str = "trend",
        outputdir: str = ".",
        rebuild: bool = True,
    ):
        """
        Save the trend coefficients to a NetCDF file.
        Loop over regions if multiple regions are present in the trend data.
        The extra_keys for the region is used only if the region is not "global".

        Args:
            diagnostic_product (str, optional): Diagnostic product tag for the filename.
                Defaults to ``'trend'``.
            outputdir (str, optional): Output directory.
            rebuild (bool, optional): Overwrite existing files.
        """
        if self.trend_coef == {}:
            self.logger.error("No trend data to save. Run compute_trend first.")
            return

        self.logger.info("Saving trend coefficients to NetCDF file")
        extra_keys = {}

        regions = list(self.trend_coef.keys())
        for region in regions:
            if self.trend_coef[region].get("AQUA_dim_mean") is not None:
                extra_keys["dim_mean"] = self.trend_coef[region].attrs["AQUA_dim_mean"]
            if region != "global":
                extra_keys["region"] = region
            super().save_netcdf(
                diagnostic=self.diagnostic_name,
                diagnostic_product=diagnostic_product,
                outputdir=outputdir,
                rebuild=rebuild,
                data=self.trend_coef[region],
                extra_keys=extra_keys,
            )
            self.logger.debug("Trend coefficients for region '%s' saved to NetCDF", region)

        self.logger.info("Trend coefficients saved to NetCDF file")
