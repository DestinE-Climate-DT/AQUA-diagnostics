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
        region: str = None,
        lon_limits: list = None,
        lat_limits: list = None,
        regions_file_path: str = None,
        vert_coord: str = None,
        diagnostic_name: str = "trends",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the Trends class.

        Args:
            model (str): Model name.
            exp (str): Experiment name.
            source (str): Data source.
            catalog (str, optional): Catalog name. Resolved by the Reader if None.
            regrid (str, optional): Target grid for regridding. No regridding if None.
            startdate (str, optional): Analysis start date.
            enddate (str, optional): Analysis end date.
            region (str, optional): Region name in the centralized regions file.
            lon_limits (list, optional): Custom longitude limits ``[lon_min, lon_max]``.
            lat_limits (list, optional): Custom latitude limits ``[lat_min, lat_max]``.
            regions_file_path (str, optional): Custom regions YAML. Defaults to the
                centralized AQUA regions file.
            vert_coord (str, optional): Name of the vertical coordinate (e.g. ``'level'``,
                ``'plev'``). If None the data is treated as 2D.
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
        self.vert_coord = vert_coord

        self.region, self.lon_limits, self.lat_limits = self._set_region(
            region=region,
            regions_file_path=regions_file_path,
            lon_limits=lon_limits,
            lat_limits=lat_limits,
        )

        self.trend_coef = None

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
        outputdir: str = "./",
        rebuild: bool = True,
        reader_kwargs: dict = {},
    ):
        """
        Run the full trend analysis workflow.

        Steps: retrieve → region selection → optional dimensional mean → trend → save.

        Args:
            var (str or list): Variable(s) to analyse.
            dim_mean (str or list, optional): Dimension(s) over which to take an areal mean
                before the trend is computed (e.g. ``['lat', 'lon']`` for a regional time series).
            outputdir (str, optional): Output directory. Defaults to ``'./'``.
            rebuild (bool, optional): Whether to overwrite existing output files. Defaults to True.
            reader_kwargs (dict, optional): Extra keyword arguments forwarded to the Reader.
        """
        self.logger.info("Starting trend analysis")
        self.retrieve(var=var, reader_kwargs=reader_kwargs)
        self.data = self._apply_region(self.data, dim_mean=dim_mean)

        self.logger.info("Computing trend coefficients")
        self.trend_coef = self.compute_trend(data=self.data)

        self.logger.info("Saving results to NetCDF")
        self.save_netcdf(outputdir=outputdir, rebuild=rebuild)
        self.logger.info("Trend analysis completed")

    def _apply_region(self, data, dim_mean=None):
        """
        Apply region selection and optional field mean to a dataset.

        Args:
            data (xr.Dataset): Input data.
            dim_mean (str or list, optional): Dimension(s) over which to compute the mean.

        Returns:
            xr.Dataset: The (possibly subset and averaged) data.
        """
        has_limits = self.lon_limits is not None or self.lat_limits is not None
        if self.region is not None or has_limits:
            label = self.region if self.region is not None else "custom limits"
            self.logger.info("Applying region selection: %s", label)
            data = self.reader.select_area(data=data, lat=self.lat_limits, lon=self.lon_limits, drop=True)
            if self.region is not None:
                data.attrs["AQUA_region"] = self.region

        if dim_mean is not None:
            self.logger.info("Averaging data over dimension(s): %s", dim_mean)
            data = self.reader.fldmean(
                data,
                dims=to_list(dim_mean),
                lat_limits=self.lat_limits,
                lon_limits=self.lon_limits,
            )
        return data

    def compute_trend(self, data: xr.Dataset) -> xr.Dataset:
        """
        Compute the linear trend coefficients along ``time`` and rescale them to per-year.

        Args:
            data (xr.Dataset): Input dataset with a ``time`` dimension.

        Returns:
            xr.Dataset: Trend coefficients (one per variable) with adjusted units.
        """
        self.logger.info("Calculating linear trend")
        trender = Trender(loglevel=self.loglevel)
        trend_data = trender.coeffs(data, dim="time", skipna=True, normalize=True)
        trend_data = trend_data.sel(degree=1)
        trend_data.attrs = data.attrs

        trend_dict = {}
        for var in data.data_vars:
            self.logger.debug("Adjusting trend for variable: %s", var)
            trend_data[var].attrs = data[var].attrs
            trend_dict[var] = self.adjust_trend_for_time_frequency(trend_data[var], data)
        trend_data = xr.Dataset(trend_dict)
        trend_data.attrs.update(data.attrs)
        if self.region is not None:
            trend_data.attrs["AQUA_region"] = self.region

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

        Args:
            diagnostic_product (str, optional): Diagnostic product tag for the filename.
                Defaults to ``'trend'``.
            outputdir (str, optional): Output directory.
            rebuild (bool, optional): Overwrite existing files.
        """
        if self.trend_coef is None:
            self.logger.error("No trend data to save. Run compute_trend first.")
            return

        self.logger.info("Saving trend coefficients to NetCDF file")
        extra_keys = {}
        if self.region is not None:
            extra_keys["region"] = self.region
        super().save_netcdf(
            diagnostic=self.diagnostic_name,
            diagnostic_product=diagnostic_product,
            outputdir=outputdir,
            rebuild=rebuild,
            data=self.trend_coef,
            extra_keys=extra_keys,
        )
        self.logger.info("Trend coefficients saved to NetCDF file")
