"""Module for computing trends using xarray."""

import pandas as pd
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.reader import Trender
from aqua.core.util import to_list
from aqua.diagnostics.base import Diagnostic
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

xr.set_options(keep_attrs=True)


class Trends(Diagnostic):
    """Class to compute trends over time.

    Attributes:
        trend_coef: Trend coefficients as an ``xr.Dataset`` for a single region
            (str, None, or one-element list), or a ``dict`` mapping region key to
            dataset when two or more regions are passed.
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
        vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
        loglevel: str = "WARNING",
    ):
        """Initialize the Trends class.

        Args:
            model (str): Climate model name.
            exp (str): Experiment name.
            source (str): Data source name.
            catalog (str, optional): Path to the data catalog.
            regrid (str, optional): Regridding method.
            startdate (str, optional): Start date for data selection.
            enddate (str, optional): End date for data selection.
            diagnostic_name (str, optional): Name of the diagnostic for filenames. Defaults to "trends".
            vert_coord (str, optional): Name of the vertical dimension coordinate. Defaults to DEFAULT_OCEAN_VERT_COORD.
            loglevel (str, optional): Logging level. Default is "WARNING".

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
        self.logger = log_configure(log_name="Trends", log_level=loglevel)
        self.diagnostic_name = diagnostic_name
        if vert_coord is None:
            vert_coord = DEFAULT_OCEAN_VERT_COORD
        self.vert_coord = vert_coord
        self.trend_coef = None

    def run(
        self,
        outputdir: str = ".",
        rebuild: bool = True,
        regions: str | list = None,
        var: list = ["thetao", "so"],
        dim_mean: type = None,
        reader_kwargs: dict = {},
    ):
        """Run the trend analysis workflow.

        Retrieves data once, computes global trend coefficients, then subsets
        and saves for each requested region.

        ``self.trend_coef`` is an ``xr.Dataset`` when a single region is requested
        (str, None, or a one-element list), and a ``dict`` of datasets when two
        or more regions are passed.

        Args:
            outputdir (str, optional): Directory to save output files. Default is current directory.
            rebuild (bool, optional): If True, rebuild existing files. Default is True.
            regions (str, list, or None, optional): Region(s) for area selection.
                None means global evaluation. Applied after trend computation.
            var (list, optional): List of variable names to analyze. Default is ['thetao', 'so'].
            dim_mean (str or list, optional): Dimension(s) over which to compute the mean. Default is None.
            reader_kwargs (dict, optional): Additional keyword arguments for the data reader. Default is {}.

        """
        self.logger.info("Starting trend analysis workflow")
        super().retrieve(var=var, reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)
        # self.data = self.data.chunk(chunks={"time": 12, "level": 1})  # this is needed to avoid a too large graph

        self.logger.info("Computing trend coefficients")
        trend_global = self.compute_trend(data=self.data)

        regions_list = to_list(regions)
        if not regions_list:
            regions_list = [None]
        as_dict = len(regions_list) > 1

        results = {}
        for reg in regions_list:
            data, region_name = self.select_region(data=trend_global, region=reg, dim_mean=dim_mean)
            self.region = region_name
            results[reg] = data

        self.trend_coef = results if as_dict else results[regions_list[0]]

        self.logger.info("Saving results to NetCDF")
        if as_dict:
            self.save_netcdf(outputdir=outputdir, rebuild=rebuild, save_all=True)
        else:
            self.save_netcdf(outputdir=outputdir, rebuild=rebuild, region=regions_list[0])

        self.logger.info("Trend analysis workflow completed")

    def select_region(self, data, region=None, drop=True, dim_mean=None):
        """Select a region and optionally compute mean over specified dimensions.

        Args:
            data (xr.Dataset): Input dataset.
            region (str, optional): Geographical region to select. None means global.
            drop (bool, optional): Whether to drop coordinates outside the region. Default is True.
            dim_mean (str or list, optional): Dimension(s) over which to compute the mean.

        Returns:
            tuple: (data, region) - Processed data and region name.

        """
        self.logger.info(
            "Processing region: %s for diagnostic '%s'.",
            region if region is not None else "global",
            self.diagnostic_name,
        )
        res_dict = super().select_region(data=data, region=region, drop=drop)
        self.region = res_dict["region"] if res_dict["region"] is not None else "global"
        self.lat_limits = res_dict["lat_limits"]
        self.lon_limits = res_dict["lon_limits"]
        if dim_mean is not None:
            self.logger.debug("Computing fldmean over dimension: %s", dim_mean)
            data = self.reader.fldmean(
                data=data,
                dims=dim_mean,
                lat_limits=self.lat_limits,
                lon_limits=self.lon_limits,
            )
        else:
            data = res_dict["data"]
        data.attrs["AQUA_region"] = self.region
        return data, self.region

    def adjust_trend_for_time_frequency(self, trend, y_array):
        """Adjust trend values based on the time frequency of the data.

        Args:
            trend (xr.DataArray): Trend values to adjust.
            y_array (xr.DataArray): Original data array with time coordinate.

        Returns:
            xr.DataArray: Adjusted trend values.

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
            trend = trend
        else:
            self.logger.error("Unsupported time frequency: %s", time_frequency)
            raise ValueError(f"The frequency: {time_frequency} of the data must be in Daily/Monthly/Yearly")

        units = trend.attrs.get("units", "")
        trend.attrs["units"] = f"{units}/year" if units else "per year"
        self.logger.debug("Trend units updated to: %s", trend.attrs["units"])
        return trend

    def compute_trend(self, data: xr.DataArray | xr.Dataset, region_name: str = None):
        """Compute linear trend coefficients over time.

        Args:
            data (xr.DataArray or xr.Dataset): Input data with a time dimension.
            region_name (str, optional): Region name stored on ``AQUA_region`` attribute.

        Returns:
            xr.DataArray or xr.Dataset: Trend coefficients adjusted for time frequency.

        """
        self.logger.info("Calculating linear trend")
        trend_init = Trender()
        trend_data = trend_init.coeffs(data, dim="time", skipna=True, normalize=True)
        trend_data = trend_data.sel(degree=1)
        trend_data.attrs = data.attrs
        trend_dict = {}
        for var in data.data_vars:
            self.logger.debug("Adjusting trend for variable: %s", var)
            trend_data[var].attrs = data[var].attrs
            trend_dict[var] = self.adjust_trend_for_time_frequency(trend_data[var], data)
        trend_data = xr.Dataset(trend_dict)
        if region_name is not None:
            trend_data.attrs["AQUA_region"] = region_name
        self.logger.info("Trend value calculated")

        self.logger.debug("Loading trend data in memory")
        trend_data.load()
        self.logger.debug("Loaded trend data in memory")
        return trend_data

    def save_netcdf(
        self,
        diagnostic_product: str = "trend",
        region: str = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_all: bool = False,
    ):
        """Save trend coefficients to NetCDF file(s).

        Args:
            diagnostic_product (str, optional): Product type for filenames. Default is "trend".
            region: Region key when ``trend_coef`` is a dict, or ignored when it is a
                single dataset. Ignored when ``save_all`` is True.
            outputdir (str, optional): Directory to save output files. Default is current directory.
            rebuild (bool, optional): If True, rebuild existing files. Default is True.
            save_all (bool): If True and ``trend_coef`` is a dict, save every entry.
                Needed because ``region=None`` is a valid dict key for global.

        """
        self.logger.info("Saving trend coefficients to NetCDF file")
        if isinstance(self.trend_coef, dict):
            if save_all:
                regions_to_save = self.trend_coef
            else:
                regions_to_save = {region: self.trend_coef[region]}
        else:
            regions_to_save = {region: self.trend_coef}

        for reg, trend_data in regions_to_save.items():
            file_region = trend_data.attrs.get("AQUA_region", reg if reg is not None else "global")
            super().save_netcdf(
                diagnostic=self.diagnostic_name,
                diagnostic_product=diagnostic_product,
                outputdir=outputdir,
                rebuild=rebuild,
                data=trend_data,
                extra_keys={"region": file_region},
            )
        self.logger.info("Trend coefficients saved to NetCDF file")
