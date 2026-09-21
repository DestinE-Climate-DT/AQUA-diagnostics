"""Module for computing trends of one or more variables along the time dimension."""

import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.reader import Trender
from aqua.core.util import to_list
from aqua.diagnostics.base import Diagnostic

xr.set_options(keep_attrs=True)

# Nanoseconds in a Julian year. The polynomial fit returns the slope per nanosecond of the
# time axis, so a single multiplication rescales the trend to per-year units whatever the
# frequency of the input data is.
NANOSECONDS_PER_YEAR = 365.25 * 24 * 3600 * 1e9


class Trends(Diagnostic):
    """
    Class to compute linear trends along the time dimension for one or more variables.

    The trend is computed through a polynomial fit and rescaled to per-year units.
    Both 2D and 3D fields are supported, the trend of a 3D field being computed for
    each vertical level.

    An instance holds a single retrieval and produces a single trend, leaving to the
    caller the choice of how to handle multiple regions:

    - without ``dim_mean`` the trend is pointwise in space and therefore commutes with
      the area selection: compute it once and slice it as many times as needed with
      :meth:`aqua.diagnostics.base.Diagnostic.select_region`.
    - with ``dim_mean`` the field mean depends on the domain, so :meth:`compute_trend`
      has to be called once per region.
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

        Args:
            model (str): Model name.
            exp (str): Experiment name.
            source (str): Data source.
            catalog (str, optional): Catalog name. Resolved by the Reader if None.
            regrid (str, optional): Target grid for regridding. No regridding if None.
            startdate (str, optional): Analysis start date.
            enddate (str, optional): Analysis end date.
            diagnostic_name (str, optional): Diagnostic name used in output filenames. Defaults to 'trends'.
            loglevel (str, optional): Logging level. Defaults to 'WARNING'.
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

        # Trend coefficients produced by the last run(), as an xr.Dataset with one variable each.
        self.trend_coef = None

    def retrieve(self, var, reader_kwargs: dict = {}):
        """
        Retrieve the data for one or more variables.

        Args:
            var (str or list): Variable name(s) to retrieve.
            reader_kwargs (dict, optional): Extra keyword arguments forwarded to the Reader.
        """
        self.logger.info("Retrieving variable(s): %s", var)
        super().retrieve(var=to_list(var), reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)

    def compute_trend(
        self,
        region: str = None,
        lon_limits: list = None,
        lat_limits: list = None,
        regions_file_path: str = None,
        dim_mean=None,
    ) -> xr.Dataset:
        """
        Compute the linear trend along ``time``, rescaled to per-year units.

        The data are optionally restricted to a region and optionally averaged over one
        or more dimensions before the fit. Can be called repeatedly on the same instance,
        the retrieved data being left untouched.

        Args:
            region (str, optional): Region name in the centralized regions file.
            lon_limits (list, optional): Custom longitude limits ``[lon_min, lon_max]``. Overridden by region.
            lat_limits (list, optional): Custom latitude limits ``[lat_min, lat_max]``. Overridden by region.
            regions_file_path (str, optional): Custom regions YAML. Defaults to the centralized AQUA regions file.
            dim_mean (str or list, optional): Dimension(s) over which to take an area-weighted mean
                before the trend is computed (e.g. ``'lon'`` for a zonal trend).

        Returns:
            xr.Dataset: Trend coefficients, one variable each, with per-year units.
        """
        if self.data is None:
            raise ValueError("No data available, run retrieve() first.")

        region, lon_limits, lat_limits = self._set_region(
            region=region,
            regions_file_path=regions_file_path,
            lon_limits=lon_limits,
            lat_limits=lat_limits,
        )

        data = self.data

        if dim_mean is not None:
            # The mean has to be applied before the fit, and it takes care of the area
            # selection itself, so that it is weighted over the region only.
            self.logger.info("Averaging data over dimension(s): %s", dim_mean)
            data = self.reader.fldmean(data, dims=to_list(dim_mean), lon_limits=lon_limits, lat_limits=lat_limits)
        elif region is not None or lon_limits is not None or lat_limits is not None:
            self.logger.info("Applying area selection: %s", region if region is not None else "custom limits")
            data = self.reader.select_area(data=data, lon=lon_limits, lat=lat_limits, drop=True)

        self.logger.info("Calculating linear trend")
        coeffs = Trender(loglevel=self.loglevel).coeffs(data, dim="time", skipna=True, normalize=False)
        trend = coeffs.sel(degree=1, drop=True) * NANOSECONDS_PER_YEAR

        # polyfit drops the coordinates which are not indexed on the fitted dimension, e.g. the
        # lat/lon defined over ncells on HealPix and other non-standard grids. Restore them.
        # TODO: this would be better placed in Trender.coeffs(), see aqua-core.
        dropped_coords = {
            name: coord for name, coord in data.coords.items() if name not in trend.coords and "time" not in coord.dims
        }
        if dropped_coords:
            self.logger.debug("Restoring coordinates dropped by polyfit: %s", list(dropped_coords))
            trend = trend.assign_coords(dropped_coords)

        trend.attrs.update(data.attrs)
        for name in trend.data_vars:
            trend[name].attrs = dict(data[name].attrs)
            units = trend[name].attrs.get("units", "")
            trend[name].attrs["units"] = f"{units}/year" if units else "per year"
        if region is not None:
            trend.attrs["AQUA_region"] = region
        if dim_mean is not None:
            trend.attrs["AQUA_dim_mean"] = "_".join(to_list(dim_mean))

        self.logger.debug("Loading trend data in memory")
        return trend.load()

    def save_netcdf(
        self,
        data: xr.Dataset = None,
        diagnostic_product: str = "trend",
        outputdir: str = ".",
        rebuild: bool = True,
    ):
        """
        Save the trend coefficients to a NetCDF file.

        Region and dimensional mean are read from the data attributes, so that a trend
        sliced afterwards with ``select_region`` is saved under its own region name.

        Args:
            data (xr.Dataset, optional): Trend coefficients to save. Defaults to ``self.trend_coef``.
            diagnostic_product (str, optional): Diagnostic product tag for the filename. Defaults to 'trend'.
            outputdir (str, optional): Output directory.
            rebuild (bool, optional): Overwrite existing files.
        """
        data = self.trend_coef if data is None else data
        if data is None:
            self.logger.error("No trend data to save, run compute_trend() first.")
            return

        extra_keys = {}
        region = data.attrs.get("AQUA_region")
        if region is not None:
            extra_keys["region"] = region
        dim_mean = data.attrs.get("AQUA_dim_mean")
        if dim_mean is not None:
            extra_keys["dim_mean"] = dim_mean

        self.logger.info("Saving trend coefficients to NetCDF file")
        super().save_netcdf(
            data=data,
            diagnostic=self.diagnostic_name,
            diagnostic_product=diagnostic_product,
            outputdir=outputdir,
            rebuild=rebuild,
            extra_keys=extra_keys,
        )

    def run(
        self,
        var,
        region: str = None,
        lon_limits: list = None,
        lat_limits: list = None,
        regions_file_path: str = None,
        dim_mean=None,
        outputdir: str = "./",
        rebuild: bool = True,
        reader_kwargs: dict = {},
    ) -> xr.Dataset:
        """
        Run the full trend analysis workflow: retrieve, compute the trend and save it.

        Args:
            var (str or list): Variable(s) to analyse.
            region (str, optional): Region name in the centralized regions file.
            lon_limits (list, optional): Custom longitude limits ``[lon_min, lon_max]``. Overridden by region.
            lat_limits (list, optional): Custom latitude limits ``[lat_min, lat_max]``. Overridden by region.
            regions_file_path (str, optional): Custom regions YAML. Defaults to the centralized AQUA regions file.
            dim_mean (str or list, optional): Dimension(s) over which to take an area-weighted mean
                before the trend is computed (e.g. ``'lon'`` for a zonal trend).
            outputdir (str, optional): Output directory. Defaults to './'.
            rebuild (bool, optional): Whether to overwrite existing output files. Defaults to True.
            reader_kwargs (dict, optional): Extra keyword arguments forwarded to the Reader.

        Returns:
            xr.Dataset: The trend coefficients, also stored in ``self.trend_coef``.
        """
        self.logger.info("Starting trend analysis")
        self.retrieve(var=var, reader_kwargs=reader_kwargs)
        self.trend_coef = self.compute_trend(
            region=region,
            lon_limits=lon_limits,
            lat_limits=lat_limits,
            regions_file_path=regions_file_path,
            dim_mean=dim_mean,
        )
        self.save_netcdf(outputdir=outputdir, rebuild=rebuild)
        self.logger.info("Trend analysis completed")
        return self.trend_coef
