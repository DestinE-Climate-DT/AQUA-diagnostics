"""Overturning tool: meridional overturning streamfunction of an ocean basin, in depth coordinates."""

import dask
import numpy as np
import xarray as xr
from scipy import sparse

from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, to_list
from aqua.diagnostics.base import Diagnostic
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

xr.set_options(keep_attrs=True)

EARTH_RADIUS = 6.371e6  # m
SVERDRUP = 1.0e6  # m3 s-1
# A grid whose latitude rows hold fewer cells than this on average is not made of iso-latitude rows
# (native curvilinear or unstructured ocean grid): the zonal integral is not defined on it.
MIN_CELLS_PER_ROW = 8


class Overturning(Diagnostic):
    """
    Compute the meridional overturning streamfunction of an ocean basin, for one dataset.

    The streamfunction is psi(y, z) = int_z^0 int_xw^xe v dx dz', in Sv: positive for northward flow above
    southward flow, as in the Atlantic overturning cell. It is computed on grids made of iso-latitude rows,
    i.e. regular lon/lat grids and HEALPix: native ocean grids must be regridded first.
    """

    # One full year: the time-mean section and at least one annual value of the index
    MINIMUM_MONTHS_REQUIRED = 12
    # Names of the NetCDF products; save_netcdf() and load() must agree on them
    PRODUCTS = ("streamfunction", "index_monthly", "index_annual")

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
        vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
        diagnostic_name: str = "overturning",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the Overturning class.

        Args:
            model (str): The model to be used.
            exp (str): The experiment to be used.
            source (str): The source to be used.
            catalog (str, optional): The catalog to be used. If None, it is determined by the Reader.
            regrid (str, optional): The target grid for regridding. If None, no regridding is done.
            startdate (str, optional): The start date of the analysis. If None, all data are used.
            enddate (str, optional): The end date of the analysis. If None, all data are used.
            region (str, optional): The basin, a region name of the regions file (e.g. 'atlantic_moc').
                If None, the global ocean is used.
            vert_coord (str, optional): The name of the vertical coordinate, in m. Default is 'depth'.
            diagnostic_name (str, optional): The name used for the output files,
                usually the collection name. Default is 'overturning'.
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
        self.logger = log_configure(log_level=loglevel, log_name="Overturning")
        self.diagnostic_name = diagnostic_name
        self.region_key = region
        self.region, _, _ = self._set_region(region=region)
        self.vert_coord = vert_coord or DEFAULT_OCEAN_VERT_COORD
        self.streamfunction = None
        self.index_monthly = None
        self.index_annual = None
        self._psi = None

    def run(
        self,
        var: str = "von",
        latitudes: list = (26.5,),
        min_depth: float = 500.0,
        lat_range: list = (-34.0, 70.0),
        mass_correction: bool = False,
        reader_kwargs: dict = None,
        outputdir: str = ".",
        rebuild: bool = True,
        save_netcdf: bool = True,
    ):
        """
        Retrieve the meridional velocity, compute the streamfunction and the index, and optionally save them.

        Args:
            var (str, optional): The northward sea water velocity. Default is 'von'.
            latitudes (list, optional): The latitudes of the overturning index. Default is [26.5] (RAPID array).
            min_depth (float, optional): The index is the maximum of the streamfunction below this depth (m),
                to leave out the wind-driven Ekman cell. Default is 500.
            lat_range (list, optional): The southern and northern limits of the section. Default is [-34, 70].
            mass_correction (bool, optional): If True, remove at each latitude the net transport across the
                section, spread uniformly over its area, so that psi vanishes at the bottom. Default is False.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_netcdf (bool, optional): If True, save the results to NetCDF. Default is True.

        Raises:
            NotEnoughDataError: If less than MINIMUM_MONTHS_REQUIRED months are available.
            ValueError: If the grid is not made of iso-latitude rows (native ocean grid).
        """
        self.retrieve(var=var, reader_kwargs=reader_kwargs or {}, months_required=self.MINIMUM_MONTHS_REQUIRED)
        self.compute_streamfunction(var=var, lat_range=lat_range, mass_correction=mass_correction)
        self.compute_index(latitudes=latitudes, min_depth=min_depth)
        if save_netcdf:
            self.save_netcdf(var=var, outputdir=outputdir, rebuild=rebuild)

    def compute_streamfunction(self, var: str = "von", lat_range: list = (-34.0, 70.0), mass_correction: bool = False):
        """
        Compute the monthly streamfunction psi(time, depth, lat) and its time mean.

        The heavy step, the zonal integral of v over the basin, reduces the data from cells to latitude rows
        lazily, chunk by chunk; everything after it works on the small (time, depth, lat) array.
        Sets self.streamfunction (time mean) and keeps the monthly streamfunction for compute_index().

        Args:
            var (str, optional): The northward sea water velocity. Default is 'von'.
            lat_range (list, optional): The southern and northern limits of the section. Default is [-34, 70].
            mass_correction (bool, optional): If True, remove the net transport across each section.
        """
        field = self._check_data(self.data[var], var=var, units="m s-1")
        if self.vert_coord not in field.dims:
            raise ValueError(f"Vertical coordinate '{self.vert_coord}' not found in {var}: dims are {field.dims}")

        # Zonal integral of v and of the wet width (for the bottom mask and the mass correction)
        transport, wet_width = self._zonal_integral(field, lat_range=lat_range)
        dz = self._layer_thickness(field[self.vert_coord])
        # The only computation on the full field: everything below is on (time, depth, lat)
        transport, wet_area = dask.compute(transport * dz, wet_width * dz)  # m3 s-1 and m2 per layer
        if transport.time.size > 1:
            transport = self.reader.timmean(transport, freq="monthly")

        net = transport.sum(self.vert_coord)
        self.logger.info("Largest net transport across a section: %.2f Sv", float(np.abs(net).max()) / SVERDRUP)
        if mass_correction:
            section_area = wet_area.sum(self.vert_coord)
            transport = transport - net * wet_area / section_area.where(section_area > 0)

        # psi at the lower interface of each layer, NaN below the deepest wet layer of each row
        psi = transport.cumsum(self.vert_coord) / SVERDRUP
        psi = psi.where(wet_area > 0)
        interfaces = self._interfaces(field[self.vert_coord].values)
        psi = psi.assign_coords({self.vert_coord: interfaces[1:]})
        psi[self.vert_coord].attrs.update({"units": "m", "long_name": "depth of the lower interface", "positive": "down"})
        psi["lat"].attrs.update({"units": "degrees_north", "long_name": "latitude"})

        psi.attrs = {k: v for k, v in self.data.attrs.items() if k.startswith("AQUA_")}
        if self.region is not None:
            psi.attrs["AQUA_region"] = self.region
        psi.attrs.update(
            {
                "units": "Sv",
                "short_name": "moc",
                "long_name": "meridional overturning streamfunction",
                "comment": "psi(y, z) = int_z^0 int v dx dz'" + (", mass corrected" if mass_correction else ""),
            }
        )
        psi.name = "moc"
        self._psi = psi

        self.streamfunction = psi.mean("time", keep_attrs=True)
        self.streamfunction.name = "moc"

    def compute_index(self, latitudes: list = (26.5,), min_depth: float = 500.0):
        """
        Compute the overturning index: the maximum of psi below min_depth at the given latitudes.

        Sets self.index_monthly and self.index_annual (dims time, lat), taken on the nearest latitude row.

        Args:
            latitudes (list, optional): The latitudes of the index. Default is [26.5].
            min_depth (float, optional): The minimum depth of the maximum (m). Default is 500.
        """
        latitudes = [float(lat) for lat in to_list(latitudes)]
        psi = self._psi.sel(lat=latitudes, method="nearest")
        for lat, row in zip(latitudes, psi["lat"].values):
            self.logger.info("Index at %g N taken on the grid row at %.3f N", lat, row)
        index = psi.where(psi[self.vert_coord] >= min_depth).max(self.vert_coord, keep_attrs=True)
        index = index.assign_coords(lat=latitudes)
        index.attrs.update({"short_name": "moc_index", "long_name": f"maximum overturning below {min_depth:g} m"})
        index.name = "moc_index"
        self.index_monthly = index

        annual = self.reader.timmean(index, freq="annual", exclude_incomplete=True)
        annual.attrs = dict(index.attrs)
        annual.name = "moc_index"
        self.index_annual = annual if annual.time.size > 0 else None

    def _zonal_integral(self, field: xr.DataArray, lat_range: list) -> tuple:
        """
        Integrate v and the wet width over the basin along each latitude row.

        Regular lon/lat grids sum over the longitude; unstructured grids made of iso-latitude rows (HEALPix)
        are reduced row by row with a sparse matrix, which keeps the operation lazy and cheap.

        Args:
            field (xr.DataArray): The meridional velocity, in m s-1.
            lat_range (list): The southern and northern limits of the section.

        Returns:
            tuple: The zonal integral of v (m2 s-1, lazy) and the wet width of each row and level (m).
        """
        if "lat" not in field.coords or "lon" not in field.coords:
            raise ValueError("lon/lat coordinates not found: the grid is not supported")
        lat = field["lat"]
        if lat.ndim != 1:
            raise ValueError("Curvilinear native grid: regrid the data to a regular or HEALPix grid")
        horizontal_dims = set(lat.dims) | set(field["lon"].dims)
        basin = self._basin_mask(field.isel({d: 0 for d in field.dims if d not in horizontal_dims}, drop=True))
        # Wet cells at every level, from the first time step
        wet = field.isel(time=0, drop=True).notnull()
        lat_min, lat_max = min(lat_range), max(lat_range)

        if "lat" in field.dims:  # regular lon/lat grid
            dlon = np.abs(np.gradient(np.unwrap(np.deg2rad(field["lon"].values))))
            dx = EARTH_RADIUS * np.cos(np.deg2rad(lat)) * xr.DataArray(dlon, dims="lon", coords={"lon": field["lon"]})
            weights = (dx * basin).where((lat >= lat_min) & (lat <= lat_max), drop=True)
            transport = (field * weights).sum("lon").sortby("lat")
            wet_width = (wet * weights).sum("lon").sortby("lat").load()
            return transport, wet_width

        # Unstructured grid: group the cells by latitude row
        cell_dim = lat.dims[0]
        lat_values = np.round(lat.values, 6)
        rows, inverse, counts = np.unique(lat_values, return_inverse=True, return_counts=True)
        if lat_values.size / rows.size < MIN_CELLS_PER_ROW:
            raise ValueError("The grid is not made of iso-latitude rows: regrid the data to a regular or HEALPix grid")
        # Width of a cell along its row: the row length shared by its cells (the grid must be global)
        dx = 2.0 * np.pi * EARTH_RADIUS * np.cos(np.deg2rad(lat_values)) / counts[inverse]
        cells = np.flatnonzero(basin.values & (lat_values >= lat_min) & (lat_values <= lat_max))
        if cells.size == 0:
            raise ValueError(f"No cell of the basin between {lat_min} and {lat_max} N")
        out_rows, row_index = np.unique(inverse[cells], return_inverse=True)
        # (cells, rows) operator: each cell contributes its width to its row
        operator = sparse.csr_matrix((dx[cells], (np.arange(cells.size), row_index)), shape=(cells.size, out_rows.size))

        def _to_rows(values):
            shape = values.shape
            flat = np.nan_to_num(values.reshape(-1, shape[-1]), nan=0.0)
            return np.asarray(flat @ operator).reshape(*shape[:-1], out_rows.size)

        def _reduce(data):
            return xr.apply_ufunc(
                _to_rows,
                data.drop_vars(["lat", "lon"], errors="ignore").isel({cell_dim: cells}),
                input_core_dims=[[cell_dim]],
                output_core_dims=[["lat"]],
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"lat": out_rows.size}, "allow_rechunk": True},
            ).assign_coords(lat=rows[out_rows])

        transport = _reduce(field)
        wet_width = _reduce(wet.astype(float)).load()
        return transport, wet_width

    def _basin_mask(self, horizontal: xr.DataArray) -> xr.DataArray:
        """
        Return the boolean mask of the basin on the horizontal grid (all True without a region).

        Args:
            horizontal (xr.DataArray): A horizontal slice of the field, with its lon/lat coordinates.
        """
        template = xr.ones_like(horizontal, dtype=float).load()
        if self.region_key is None:
            return template.astype(bool)
        selected = self.select_region(template, region=self.region_key, drop=False, to_180=False)["data"]
        mask = selected.notnull()
        if not mask.any():
            raise ValueError(f"The basin {self.region} has no cell on this grid")
        return mask

    @staticmethod
    def _interfaces(depth: np.ndarray) -> np.ndarray:
        """
        Return the layer interfaces from the layer centres: the surface, the midpoints, and a bottom interface
        as far below the last centre as the last midpoint is above it.

        Args:
            depth (np.ndarray): The depth of the layer centres (m, positive down, increasing).
        """
        depth = np.asarray(depth, dtype=float)
        if depth.size == 1:
            return np.array([0.0, 2.0 * depth[0]])
        mid = 0.5 * (depth[1:] + depth[:-1])
        return np.concatenate([[0.0], mid, [2.0 * depth[-1] - mid[-1]]])

    def _layer_thickness(self, depth: xr.DataArray) -> xr.DataArray:
        """
        Return the layer thickness (m) on the vertical coordinate.

        Args:
            depth (xr.DataArray): The vertical coordinate, depth of the layer centres in m.
        """
        values = depth.values.astype(float)
        if np.any(np.diff(values) <= 0) or values[0] < 0:
            raise ValueError(f"'{self.vert_coord}' must be positive and increasing downwards (m)")
        return xr.DataArray(np.diff(self._interfaces(values)), dims=depth.dims, coords=depth.coords)

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

    def save_netcdf(self, var: str = "von", outputdir: str = ".", rebuild: bool = True):
        """
        Save the streamfunction and the indices to NetCDF files, one per product.

        Args:
            var (str, optional): The variable name in the filename. Default is 'von'.
            outputdir (str, optional): The output directory. Default is '.'.
            rebuild (bool, optional): If True, overwrite an existing file. Default is True.
        """
        for product in self.PRODUCTS:
            result = getattr(self, product)
            if result is None:
                continue
            # load_netcdf() matches startdate/enddate against the dataset-level attributes, while a bare
            # DataArray keeps them on the variable: store a Dataset with the attributes at both levels.
            data = result.to_dataset()
            data.attrs.update(result.attrs)
            super().save_netcdf(
                data=data,
                diagnostic=self.diagnostic_name,
                diagnostic_product=product,
                outputdir=outputdir,
                rebuild=rebuild,
                extra_keys=self._extra_keys(var),
            )

    def load(self, var: str = "von", outputdir: str = ".", reader_kwargs: dict = None):
        """
        Populate the results from the NetCDF files written by a previous run.

        A missing file, or one computed over a different period, leaves the result untouched.

        Args:
            var (str, optional): The variable that was analysed. Default is 'von'.
            outputdir (str, optional): The directory where the data was saved. Default is '.'.
            reader_kwargs (dict, optional): The Reader keyword arguments of the run, to match its realization.
        """
        self.realization = (reader_kwargs or {}).get("realization", DEFAULT_REALIZATION)
        for product in self.PRODUCTS:
            data = self.load_netcdf(
                diagnostic=self.diagnostic_name,
                diagnostic_product=product,
                outputdir=outputdir,
                extra_keys=self._extra_keys(var),
                as_dataarray=True,
                startdate=self.startdate,
                enddate=self.enddate,
            )
            if data is not None:
                setattr(self, product, data)
