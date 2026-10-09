"""Module for computing Hovmoller diagrams from ocean drift diagnostics."""

from itertools import product

import dask
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, to_list
from aqua.diagnostics.base import Diagnostic
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

xr.set_options(keep_attrs=True)


class Hovmoller(Diagnostic):
    """A class for generating Hovmoller diagrams from ocean model data.

    This class provides methods to retrieve, process, and save netCDF files
    for Hovmoller diagrams. It inherits from the `Diagnostic` class.

    Attributes:
        logger (Logger): Logger instance for the class.
        processed_data (dict): Mapping of region name to list of processed datasets
            (anomaly/standardise combinations).

    """

    MINIMUM_MONTHS_REQUIRED = 2

    def __init__(
        self,
        model: str,
        exp: str,
        source: str,
        catalog: str = None,
        regrid: str = None,
        startdate: str = None,
        enddate: str = None,
        diagnostic_name: str = "oceandrift",
        vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
        loglevel: str = "WARNING",
    ):
        """Initialize the Hovmoller class.

        Args:
            model (str): Model name.
            exp (str): Experiment name.
            source (str): Data source.
            catalog (str, optional): Path to the catalog file.
            regrid (str, optional): Regridding method.
            startdate (str, optional): Start date for data retrieval.
            enddate (str, optional): End date for data retrieval.
            diagnostic_name (str, optional): Name of the diagnostic for filenames. Defaults to "oceandrift".
            vert_coord (str, optional): Name of the vertical dimension coordinate. Defaults to DEFAULT_OCEAN_VERT_COORD.
            loglevel (str, optional): Logging level. Defaults to "WARNING".

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
        self.logger = log_configure(log_name="OceanHovmoller", log_level=loglevel)
        self.diagnostic_name = diagnostic_name
        if vert_coord is None:
            vert_coord = DEFAULT_OCEAN_VERT_COORD
        self.vert_coord = vert_coord
        self.processed_data = {}

    def run(
        self,
        outputdir: str = ".",
        rebuild: bool = True,
        regions: str | list = None,
        var: list = ["thetao", "so"],
        dim_mean=["lat", "lon"],
        anomaly_ref: str = None,
        reader_kwargs: dict = {},
        save_netcdf: bool = True,
    ):
        """Run the Hovmoller diagram generation workflow.

        Retrieves data once. With a spatial mean, each year is reduced once and every
        region is averaged on that year in a single scheduler pass. Hovmoller products
        are built from the concatenated time series.

        Args:
            outputdir (str, optional): Directory to save the output files. Defaults to ".".
            rebuild (bool, optional): Whether to rebuild the netCDF file. Defaults to True.
            regions (str, list, or None, optional): Region(s) for area selection.
                None means global evaluation. A list processes each region with a single retrieve.
            var (list, optional): List of variables to process. Defaults to ["thetao", "so"].
            dim_mean (list, optional): List of dimensions over which to compute the mean. Defaults to ["lat", "lon"].
            anomaly_ref (str or None, optional): Reference for anomaly calculation. Can be "t0", "tmean", or None.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader. Defaults to {}.
            save_netcdf (bool, optional): Whether to save the processed data as netcdf files. Defaults to True.

        """
        self.logger.info("Running Hovmoller diagram generation")
        super().retrieve(var=var, reader_kwargs=reader_kwargs, months_required=self.MINIMUM_MONTHS_REQUIRED)
        self._fix_vert_coord_units()

        retrieved_data = self.data
        regions_list = to_list(regions)
        if not regions_list:
            regions_list = [None]

        self.processed_data = {}
        self.logger.debug("Variables retrieved: %s, regions: %s, dim_mean: %s", var, regions_list, dim_mean)
        region_info = self._region_limits(regions_list)
        # Invalid regions are dropped by _region_limits, so they do not stop the valid ones
        regions_list = list(region_info)
        if not regions_list:
            raise ValueError(f"None of the requested regions {to_list(regions)} is defined in the regions file")

        if dim_mean is not None:
            region_means = self._fldmean_regions_by_year(
                data=retrieved_data,
                regions_list=regions_list,
                region_info=region_info,
                dim_mean=dim_mean,
            )
        else:
            region_means = None

        for reg in regions_list:
            info = region_info[reg]
            self.region = info["region_name"]
            self.lat_limits = info["lat_limits"]
            self.lon_limits = info["lon_limits"]
            self.logger.info(
                "Processing region: %s for diagnostic '%s'.",
                self.region,
                self.diagnostic_name,
            )
            if region_means is not None:
                data = region_means[reg]
            else:
                data = super().select_region(data=retrieved_data, region=reg, drop=True)["data"]
            self.processed_data[reg] = self.compute_hovmoller(
                data=data,
                anomaly_ref=anomaly_ref,
                region_name=self.region,
            )
            if save_netcdf:
                self.save_netcdf(outputdir=outputdir, rebuild=rebuild, region=reg)

        self.logger.info("Hovmoller diagram computation completed")

    def load(
        self,
        diagnostic_product: str = "hovmoller",
        outputdir: str = ".",
        regions: str | list = None,
        anomaly_ref: str | list = None,
        reader_kwargs: dict = {},
    ):
        """Populate the processed data from the netcdf files written by a previous run.

        Regions with no files on disk are left untouched, so that load can be called both before run,
        to keep previous results, and after it, or alone to plot without recomputing anything.

        Args:
            diagnostic_product (str, optional): Name of the diagnostic product. Defaults to "hovmoller".
            outputdir (str, optional): Directory where the data was saved. Defaults to ".".
            regions (str, list, or None, optional): Region(s) given to run. None means global evaluation.
            anomaly_ref (str, list or None, optional): Reference for anomaly calculation given to run.
            reader_kwargs (dict, optional): The Reader keyword arguments of the run, to match its realization.

        """
        self.realization = reader_kwargs.get("realization", DEFAULT_REALIZATION)
        drift_types = [self._drift_type(ref, do_standardise) for do_standardise, ref in self._drift_products(anomaly_ref)]

        regions_list = to_list(regions)
        if not regions_list:
            regions_list = [None]

        # The region names used in the filenames are resolved without any data access
        for reg, info in self._region_limits(regions_list).items():
            processed = []
            for drift_type in drift_types:
                data = self.load_netcdf(
                    diagnostic=self.diagnostic_name,
                    diagnostic_product=diagnostic_product,
                    outputdir=outputdir,
                    extra_keys=self._extra_keys(region_name=info["region_name"], drift_type=drift_type),
                )
                if data is not None:
                    processed.append(data)

            # A partial set would silently plot fewer rows than a run does
            if len(processed) < len(drift_types):
                self.logger.info("Some Hovmoller products are missing for %s, nothing loaded", info["region_name"])
                continue
            self.processed_data[reg] = sorted(processed, key=self.sort_drift_type)

    def _region_limits(self, regions_list: list) -> dict:
        """Look up each region's name and latitude/longitude box once."""
        known_regions = self._load_regions_from_file()
        info = {}
        for reg in regions_list:
            if reg is not None and reg not in known_regions:
                self.logger.error("Skipping region %s: not found in the regions file", reg)
                continue
            long_name, lon_limits, lat_limits = self._set_region(region=reg)
            info[reg] = {
                "region_name": long_name if long_name is not None else "global",
                "lon_limits": lon_limits,
                "lat_limits": lat_limits,
            }
        return info

    def _fldmean_regions_by_year(self, data: xr.Dataset, regions_list: list, region_info: dict, dim_mean: list) -> dict:
        """Average every region over one year at a time, in a single scheduler pass.

        The year slice stays lazy. All region means share one ``xr.compute`` so
        each chunk is read once and discarded after the reductions. Concatenating
        the yearly means rebuilds the full time series without holding a year.
        """
        means = {reg: [] for reg in regions_list}
        years = sorted({int(year) for year in data.time.dt.year.values})
        for year in years:
            self.logger.info("Computing year %s", year)
            block = data.isel(time=(data.time.dt.year == year).values)
            lazy_means = []
            for reg in regions_list:
                info = region_info[reg]
                self.logger.debug(
                    "Computing fldmean over %s for region %s, year %s",
                    dim_mean,
                    info["region_name"],
                    year,
                )
                lazy_means.append(
                    self.reader.fldmean(
                        data=block,
                        dims=dim_mean,
                        lat_limits=info["lat_limits"],
                        lon_limits=info["lon_limits"],
                    )
                )
            computed = dask.compute(*lazy_means)
            for reg, result in zip(regions_list, computed):
                means[reg].append(result)
        return {reg: xr.concat(parts, dim="time") for reg, parts in means.items()}

    def get_anomaly(self, data: xr.DataArray, anomaly_ref: str = None, dim: str = "time") -> xr.DataArray:
        """Compute anomaly for the given data along a specified dimension.

        Args:
            data: The input data array to process.
            anomaly_ref: Reference for anomaly calculation. Can be "t0", "tmean", or None.
                If "t0" or "tmean", the anomaly is computed relative to the initial time or the mean.
                If None, no anomaly is computed.
            dim: The dimension along which to compute the anomaly. Default is "time".

        Returns:
            The anomaly data array, or the original data if anomaly_ref is None.

        """
        if anomaly_ref is None:
            return data
        if anomaly_ref == "tmean":
            return data - data.mean(dim=dim)
        if anomaly_ref == "t0":
            return data - data.isel({dim: 0})
        raise ValueError("Invalid anomaly_ref: use 't0', 'tmean', or None")

    def standardise(self, data: xr.DataArray, dim: str = "time") -> xr.DataArray:
        """Standardise the data along a specified dimension.

        Args:
            data: The input data array to standardise.
            dim: The dimension along which to standardise. Default is "time".

        Returns:
            The standardised data array with updated attributes.

        """
        data = data / data.std(dim=dim)
        data.attrs["units"] = "Stand. Units"
        data.attrs["AQUA_standardise"] = f"Standardised with {dim}"
        return data

    def apply_std_anomaly(
        self,
        data: xr.DataArray,
        anomaly_ref: str = None,
        do_standardise: bool = False,
        dim: str = "time",
        region_name: str = None,
    ) -> xr.DataArray:
        """Compute anomaly and/or standardised anomaly along a dimension.

        Args:
            data: The input data array to process.
            anomaly_ref: Reference for anomaly calculation. Can be "t0", "tmean", or None.
            do_standardise: If True, standardise the (anomaly) data.
            dim: Dimension for anomaly and/or standardisation. Default is "time".
            region: Region name stored on ``AQUA_region`` attribute.

        Returns:
            Processed data with ``AQUA_ocean_drift_type`` (and optional region) attributes set.

        """
        if anomaly_ref is not None:
            if anomaly_ref in ["t0", "tmean"]:
                data = self.get_anomaly(data, anomaly_ref, dim)
        if do_standardise:
            data = self.standardise(data, dim)

        # Shallow copy so AQUA_* attrs do not mutate the shared fldmean result
        # (needed when anomaly_ref is None and no new array was created).
        data = data.copy(deep=False)

        data.attrs["AQUA_ocean_drift_type"] = self._drift_type(anomaly_ref, do_standardise)
        if region_name is not None:
            data.attrs["AQUA_region"] = region_name
        return data

    def compute_hovmoller(
        self,
        data: xr.Dataset = None,
        anomaly_ref: str | list = None,
        region_name: str = None,
    ) -> list:
        """Process data for drift analysis by applying transforms and aggregations.

        Args:
            data: Input dataset. Defaults to ``self.data``.
            dim_mean: Dimensions along which to compute the field mean.
                If None, no mean is computed.
            anomaly_ref: Reference for anomaly calculation. Can be "t0", "tmean",
                a list of those, or None. Full (non-anomaly) values are always included.
            lat_limits: Latitude limits for fldmean. Defaults to None.
            lon_limits: Longitude limits for fldmean. Defaults to None.
            region: Region name stored on output attrs. Defaults to None.

        Returns:
            Sorted list of processed datasets for each anomaly/standardise combination.

        """
        if data is None:
            data = self.data

        processed = []
        for do_standardise, ref in self._drift_products(anomaly_ref):
            self.logger.info("Processing data with standardise=%s, anomaly_ref=%s", do_standardise, ref)
            processed.append(
                self.apply_std_anomaly(
                    data,
                    anomaly_ref=ref,
                    do_standardise=do_standardise,
                    dim="time",
                    region_name=region_name,
                )
            )
        return sorted(processed, key=self.sort_drift_type)

    def _drift_products(self, anomaly_ref: str | list = None) -> list:
        """List the (do_standardise, anomaly_ref) combinations computed for the given anomaly references.

        Full values are always included, and only anomalies are standardised.
        Shared by compute_hovmoller and load, so that the two always address the same products.
        """
        refs = to_list(anomaly_ref) + [None]
        return [(std, ref) for std, ref in product([False, True], refs) if not (std and ref is None)]

    def _drift_type(self, anomaly_ref: str = None, do_standardise: bool = False) -> str:
        """Name a drift product, as stored in the AQUA_ocean_drift_type attribute and in the filenames."""
        s_std = "std_" if do_standardise else ""
        anom = "anom" if anomaly_ref is not None else "full"
        anom_ref = f"_{anomaly_ref}" if anomaly_ref else ""
        return f"{s_std}{anom}{anom_ref}"

    def _extra_keys(self, region_name: str, drift_type: str) -> dict:
        """Build the filename keys identifying one product.

        Shared by save_netcdf and load, so that the two always address the same file.
        """
        return {"region": region_name, "ocean_drift_type": drift_type}

    def sort_drift_type(self, data) -> tuple:
        """Return a sort key for ordering processed data by drift type."""
        drift_type = data.attrs["AQUA_ocean_drift_type"]
        if drift_type == "full":
            return (0, drift_type)
        if drift_type.startswith("anom"):
            return (1, drift_type)
        if drift_type.startswith("std"):
            return (2, drift_type)
        return (3, drift_type)

    def save_netcdf(
        self,
        diagnostic_product: str = "hovmoller",
        region: str = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_all: bool = False,
    ):
        """Save processed data to netCDF files.

        Args:
            diagnostic_product (str): Name of the diagnostic product.
            region: Key in ``self.processed_data`` (same as config region id, or
                ``None`` for global). Ignored when ``save_all`` is True.
            outputdir (str): Directory to save the output files. Defaults to '.'.
            rebuild (bool, optional): Whether to rebuild the netCDF file. Defaults to True.
            save_all (bool): If True, save every entry in ``self.processed_data``.
                Needed because ``region=None`` is a valid dict key for global.

        """
        if save_all:
            regions_to_save = self.processed_data
        else:
            regions_to_save = {region: self.processed_data[region]}

        for reg, processed_list in regions_to_save.items():
            for processed_data in processed_list:
                # Prefer long name on attrs for filenames; fall back to dict key.
                file_region = processed_data.attrs.get("AQUA_region", reg if reg is not None else "global")
                super().save_netcdf(
                    data=processed_data,
                    diagnostic=self.diagnostic_name,
                    diagnostic_product=f"{diagnostic_product}",
                    outputdir=outputdir,
                    rebuild=rebuild,
                    extra_keys=self._extra_keys(
                        region_name=file_region, drift_type=processed_data.attrs["AQUA_ocean_drift_type"]
                    ),
                )

    def _fix_vert_coord_units(self):
        """Normalize vertical coordinate units and validate they are in metres."""
        # HACK: some LRA datasets have levels in 'NEMO model layers' (also non NEMO models due to multi-IO)
        if self.data[self.vert_coord].attrs["units"] == "NEMO model layers":
            self.data[self.vert_coord].attrs["units"] = "m"
        super()._check_data(data=self.data[self.vert_coord], var=self.vert_coord, units="m")
        self.logger.debug("Data retrieved successfully")
