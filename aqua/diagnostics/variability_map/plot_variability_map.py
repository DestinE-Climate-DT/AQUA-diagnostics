import numpy as np
import xarray as xr

from aqua import Regridder
from aqua.core.fldstat import AreaSelection
from aqua.core.graphics import plot_single_map, plot_single_map_diff
from aqua.core.logger import log_configure
from aqua.core.util import get_projection
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

xr.set_options(keep_attrs=True)


class PlotVariabilityMap:
    """
    Plot variability (STD) maps and the difference of variability maps between data and reference

    Note: Variability means STD in this diagnostic
    """

    def __init__(
        self,
        diagnostic_name="variability_map",
        outputdir="./",
        loglevel="WARNING",
    ):
        """
        Initialize the PlotVariabilityMap.

        Args:
            diagnostic_name (str, optional): Name used for the output files, usually the collection name.
                Default is 'variability_map'.
            outputdir (str, optional): Output directory for saved plots. Default is './'.
            loglevel (str, optional): Logging level. Default is 'WARNING'.
        """

        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotVariabilityMap")

    def plot(
        self,
        var=None,
        dataset_std=None,
        catalog=None,
        model=None,
        exp=None,
        startdate=None,
        enddate=None,
        plot_options={},
        figsize: tuple = (11, 8.5),
        ax_pos: tuple = (1, 1, 1),
        vmin=None,
        vmax=None,
        gridlines=False,
        contour=False,
        proj="robinson",
        proj_params={},
        save_format=SAVE_FORMAT,
        dpi=600,
        region=None,
        lon_limits=None,
        lat_limits=None,
        # Masking flags and boundary latitudes, specific to ICON
        mask_northern_boundary=True,
        mask_southern_boundary=True,
        northern_boundary_latitude=70,
        southern_boundary_latitude=-62,
        diagnostic_product="variability_map",
        rebuild: bool = True,
        tgt_grid_name="r1440x721",
        regrid_method="ycon",
    ):
        """
        Visualize the Variability map.

        Plot the variability map from an input dataset.

        This function visualizes variability map using configurable spatial, temporal,
        and plotting options. It supports contour, regional selection, custom projections,
        masking, and output saving in multiple formats.

        Args:
            var (str, optional): Variable to be plotted. Default is 'None'.
            dataset_std (xarray.Dataset, optional): Dataset containing the 2D fields to be plotted.
            catalog (str, optional): Catalog name. Used in plot titles. (Mandatory for labeling)
            model (str, optional): Model or dataset name. Used in plot titles. (Mandatory for labeling)
            exp (str, optional): Experiment identifier. Used in plot titles. (Mandatory for labeling)
            startdate (str, optional): Start date label to include in the plot title.
            enddate (str, optional): End date label to include in the plot title.
            plot_options (dict, optional): Additional keyword arguments for customizing the plot (e.g., colormap, linewidth).
            figsize (tuple, optional): Size of the figure. Default is (11, 8.5).
            ax_pos (tuple, optional): Position of the axes. Default is (1, 1, 1).
            vmin (float, optional): Minimum value for color scaling. If ``None``, determined automatically.
            vmax (float, optional): Maximum value for color scaling. If ``None``, determined automatically.
            gridlines (bool, optional): Whether to draw gridlines. Default is False.
            contour (bool, optional): Whether to plot using filled contours. Default is False.
            proj (str, optional): Map projection type. Default is ``'robinson'``.
            proj_params (dict, optional): Additional keyword arguments passed to the projection.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            dpi (int, optional): Resolution (dots per inch) for saved figures. Default is ``600``.
            region (str, optional): Region identifier. If provided, overrides lat/lon limits.
            lon_limits (list[float], optional): Longitude limits [min, max] for the plot.
            lat_limits (list[float], optional): Latitude limits [min, max] for the plot.
            mask_northern_boundary (bool, optional): If ``True``, mask latitudes north of ``northern_boundary_latitude``.
            mask_southern_boundary (bool, optional): If ``True``, mask latitudes south of ``southern_boundary_latitude``.
            northern_boundary_latitude (float, optional): Latitude above which data will be masked. Default is ``70``.
            southern_boundary_latitude (float, optional): Latitude below which data will be masked. Default is ``-62``.
            diagnostic_product (str, optional): Product name in the filename. Default is ``'variability_map'``.
            rebuild (bool, optional): If ``True``, rebuild the data from the original files. Default is ``True``.
            tgt_grid_name (str, optional): Target grid name for regridding. Default is 'r1440x721'.
            regrid_method (str, optional): Regridding method to use. Default is 'ycon'.

        Returns:
            matplotlib.figure.Figure: The generated plot figure object.
            matplotlib.axes.Axes: The axes object containing the plot.

        Raises:
            ValueError: If required arguments (e.g., ``catalog``, ``model``, ``exp``) are missing.
            TypeError: If inputs are of invalid type (e.g., dataset not an xarray.Dataset).
        """

        if dataset_std is None:
            raise ValueError("Please provide the data to the plot function")

        if isinstance(dataset_std, xr.Dataset):
            dataset_std = dataset_std[var]
        var = var or dataset_std.attrs.get("short_name", dataset_std.name)

        # This is important to provide the start and end dates. These dates will be used in the title of the plot
        if startdate is None or enddate is None:
            self.logger.error("Please specify the time period of the data")

        self.logger.info(f"Plotting variability (STD) map for {model} and {exp}, from {startdate} to {enddate}.")
        long_name = dataset_std.attrs.get("long_name", var)

        title = TitleBuilder(
            diagnostic="Variability Map",
            variable=long_name,
            regions=region,
            model=model,
            exp=exp,
            startyear=startdate,
            endyear=enddate,
        ).generate()

        description = f"Variability map of {long_name} for {model} {exp} ({startdate} to {enddate}) "

        if tgt_grid_name is not None:
            dataset_std = self._regrid(dataset_std, tgt_grid_name, regrid_method)

        if region is not None:
            if lon_limits is None or lat_limits is None:
                self.logger.error(f"For the {region}, please specify the lon_limits and lat_limits.")
            description = description + f"for {region} "
            dataset_std = self.subregion_selection(
                data=dataset_std,
                model=model,
                exp=exp,
                mask_northern_boundary=mask_northern_boundary,
                northern_boundary_latitude=northern_boundary_latitude,
                mask_southern_boundary=mask_southern_boundary,
                southern_boundary_latitude=southern_boundary_latitude,
                lon_lim=lon_limits,
                lat_lim=lat_limits,
                region_name=region,
            )

        if vmin is None or (isinstance(vmin, (float, int)) and np.isnan(vmin)):
            vmin = float(dataset_std.min(skipna=True))
        if vmax is None or (isinstance(vmax, (float, int)) and np.isnan(vmax)):
            vmax = float(dataset_std.max(skipna=True))

        if vmin == vmax:
            # Constant field: let plot_single_map choose the limits
            self.logger.info("STD is the same everywhere")
            vmin = vmax = None

        fig, ax = plot_single_map(
            data=dataset_std,
            contour=contour,
            return_fig=True,
            title=title,
            vmin=vmin,
            vmax=vmax,
            proj=get_projection(proj, **proj_params),
            add_land=True,
            gridlines=gridlines,
            loglevel=self.loglevel,
            figsize=figsize,
            ax_pos=ax_pos,
            **plot_options,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        self.save_plot(
            fig,
            diagnostic_product=diagnostic_product,
            description=description,
            var=var,
            region=region,
            catalog=catalog,
            model=model,
            exp=exp,
            save_format=save_format,
            dpi=dpi,
            rebuild=rebuild,
        )

        return fig, ax

    def plot_diff(
        self,
        var=None,
        dataset_std=None,
        catalog=None,
        model=None,
        exp=None,
        startdate=None,
        enddate=None,
        dataset_std_ref=None,
        catalog_ref=None,
        model_ref=None,
        exp_ref=None,
        startdate_ref=None,
        enddate_ref=None,
        figsize: tuple = (11, 8.5),
        ax_pos: tuple = (1, 1, 1),
        plot_options={},
        vmin_diff=None,
        vmax_diff=None,
        gridlines=False,
        contour=False,
        proj="robinson",
        proj_params={},
        save_format=SAVE_FORMAT,
        dpi=600,
        region=None,
        lon_limits=None,
        lat_limits=None,
        # Masking flags and boundary latitudes, specific to ICON
        mask_northern_boundary=True,
        mask_southern_boundary=True,
        northern_boundary_latitude=70,
        southern_boundary_latitude=-62,
        diagnostic_product="variability_map_difference",
        rebuild: bool = True,
        tgt_grid_name="r1440x721",
        regrid_method="ycon",
    ):
        """
        Visualize the difference in variability between a model and a reference dataset.

        This function generates a map of variability (STD) differences using Cartopy projections,
        supporting custom contour, masking, regional selection, and configurable plotting options.
        The plot can be saved as PNG or PDF.

        Args:
            var (str, optional): Variable name to plot (e.g., 'zos').
            dataset_std (xarray.Dataset, optional): Dataset of the model to be plotted.
            catalog (str, optional): Catalog name for the model dataset (used in plot title).
            model (str, optional): Model name of the dataset (used in plot title).
            exp (str, optional): Experiment name of the dataset (used in plot title).
            startdate (str, optional): Start date of the dataset for the plot title.
            enddate (str, optional): End date of the dataset for the plot title.
            dataset_std_ref (xarray.Dataset, optional): Reference dataset for comparison.
            catalog_ref (str, optional): Catalog name for the reference dataset.
            model_ref (str, optional): Model name of the reference dataset.
            exp_ref (str, optional): Experiment name of the reference dataset.
            startdate_ref (str, optional): Start date of the reference dataset.
            enddate_ref (str, optional): End date of the reference dataset.
            figsize (tuple, optional): Size of the figure. Default is (11, 8.5).
            ax_pos (tuple, optional): Position of the axes. Default is (1, 1, 1).
            plot_options (dict, optional): Additional keyword arguments for plotting (e.g., colormap, alpha).
            vmin_diff (float, optional): Minimum value for color scaling. If None, determined automatically.
            vmax_diff (float, optional): Maximum value for color scaling. If None, determined automatically.
            gridlines (bool, optional): Whether to draw gridlines. Default is False.
            contour (bool, optional): Whether to plot using filled contours. Default is False.
            proj (str, optional): Map projection. Default is 'robinson'.
            proj_params (dict, optional): Additional keyword arguments for the projection.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            dpi (int, optional): Resolution of the saved figure. Default is 600.
            region (str, optional): Region identifier for the plot.
            lon_limits (list[float], optional): Longitude limits [min, max] for the plot.
            lat_limits (list[float], optional): Latitude limits [min, max] for the plot.
            mask_northern_boundary (bool, optional): Mask latitudes above northern_boundary_latitude. Default is True.
            mask_southern_boundary (bool, optional): Mask latitudes below southern_boundary_latitude. Default is True.
            northern_boundary_latitude (float, optional): Latitude above which data is masked. Default is 70.
            southern_boundary_latitude (float, optional): Latitude below which data is masked. Default is -62.
            diagnostic_product (str, optional): Diagnostic product identifier. Default is 'variability_map_difference'.
            rebuild (bool, optional): If ``True``, rebuild the data from the original files. Default is ``True``.
            tgt_grid_name (str, optional): Target grid name for regridding. Default is 'r1440x721'.
            regrid_method (str, optional): Regridding method to use. Default is 'ycon'.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
            matplotlib.axes.Axes: The axes object containing the plot.

        Raises:
            ValueError: If required dataset or catalog/model/exp information is missing.
            TypeError: If input datasets are not xarray.Datasets.
        """
        if dataset_std is None or dataset_std_ref is None:
            raise ValueError("Please provide both the data and the reference to the plot function")

        if isinstance(dataset_std, xr.Dataset):
            dataset_std = dataset_std[var]
        if isinstance(dataset_std_ref, xr.Dataset):
            dataset_std_ref = dataset_std_ref[var]
        var = var or dataset_std.attrs.get("short_name", dataset_std.name)

        # Differences need both maps on the same grid
        if tgt_grid_name is not None:
            dataset_std = self._regrid(dataset_std, tgt_grid_name, regrid_method)
            dataset_std_ref = self._regrid(dataset_std_ref, tgt_grid_name, regrid_method)

        if startdate is None or enddate is None:
            self.logger.error("Please specify the time period of the data")
        if startdate_ref is None or enddate_ref is None:
            self.logger.error("Please specify the time period of the reference data")

        long_name = dataset_std.attrs.get("long_name", var)

        title = TitleBuilder(
            diagnostic="Variability",
            variable=long_name,
            regions=region,
            model=model,
            exp=exp,
            startyear=startdate,
            endyear=enddate,
            comparison="minus",
            ref_model=model_ref,
            ref_exp=exp_ref,
            ref_startyear=startdate_ref,
            ref_endyear=enddate_ref,
        ).generate()

        description = (
            f"Difference between Variability of {long_name} for {model} {exp} "
            f"({startdate}-{enddate}) and, reference {catalog_ref} {model_ref} and {exp_ref} "
            f"({startdate_ref}-{enddate_ref}) "
        )

        if region:
            if lon_limits is None or lat_limits is None:
                self.logger.error(f"For the {region}, please specify the lon_limits and lat_limits.")
            description = description + f"for {region} "
            dataset_std = self.subregion_selection(
                data=dataset_std,
                model=model,
                exp=exp,
                mask_northern_boundary=mask_northern_boundary,
                northern_boundary_latitude=northern_boundary_latitude,
                mask_southern_boundary=mask_southern_boundary,
                southern_boundary_latitude=southern_boundary_latitude,
                lon_lim=lon_limits,
                lat_lim=lat_limits,
                region_name=region,
            )
            dataset_std_ref = self.subregion_selection(
                data=dataset_std_ref,
                model=model_ref,
                exp=exp_ref,
                mask_northern_boundary=mask_northern_boundary,
                northern_boundary_latitude=northern_boundary_latitude,
                mask_southern_boundary=mask_southern_boundary,
                southern_boundary_latitude=southern_boundary_latitude,
                lon_lim=lon_limits,
                lat_lim=lat_limits,
                region_name=region,
            )

        diff_map = dataset_std - dataset_std_ref
        if vmin_diff is None or (isinstance(vmin_diff, (float, int)) and np.isnan(vmin_diff)):
            vmin_diff = float(diff_map.min(skipna=True))
        if vmax_diff is None or (isinstance(vmax_diff, (float, int)) and np.isnan(vmax_diff)):
            vmax_diff = float(diff_map.max(skipna=True))
        if vmin_diff == vmax_diff:
            # Constant difference (e.g. identical maps): let plot_single_map_diff choose the limits
            self.logger.warning("The difference is the same everywhere (%s)", vmin_diff)
            vmin_diff = vmax_diff = None

        fig, ax = plot_single_map_diff(
            data=dataset_std,
            data_ref=dataset_std_ref,
            contour=contour,
            return_fig=True,
            title=title,
            vmin_fill=vmin_diff,
            vmax_fill=vmax_diff,
            add_land=True,
            proj=get_projection(proj, **proj_params),
            figsize=figsize,
            ax_pos=ax_pos,
            gridlines=gridlines,
            loglevel=self.loglevel,
            **plot_options,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        self.save_plot(
            fig,
            diagnostic_product=diagnostic_product,
            description=description,
            var=var,
            region=region,
            catalog=catalog,
            model=model,
            exp=exp,
            catalog_ref=catalog_ref,
            model_ref=model_ref,
            exp_ref=exp_ref,
            save_format=save_format,
            dpi=dpi,
            rebuild=rebuild,
        )
        return fig, ax

    def save_plot(
        self,
        fig,
        diagnostic_product: str,
        description: str,
        var: str,
        region: str = None,
        catalog: str = None,
        model: str = None,
        exp: str = None,
        catalog_ref: str = None,
        model_ref: str = None,
        exp_ref: str = None,
        save_format=SAVE_FORMAT,
        dpi: int = 600,
        rebuild: bool = True,
    ):
        """
        Save a figure through the OutputSaver, with the description in its metadata.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            diagnostic_product (str): Product name in the filename.
            description (str): Caption stored in the figure metadata.
            var (str): Variable name in the filename.
            region (str, optional): Region name in the filename. 'global' if None.
            catalog, model, exp (str, optional): Dataset identifiers.
            catalog_ref, model_ref, exp_ref (str, optional): Reference identifiers, for difference plots.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            dpi (int, optional): Resolution of the saved figure. Default is 600.
            rebuild (bool, optional): If True, overwrite an existing file. Default is True.
        """
        outputsaver = OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=catalog,
            model=model,
            exp=exp,
            catalog_ref=catalog_ref,
            model_ref=model_ref,
            exp_ref=exp_ref,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
        )
        region = region.replace(" ", "").lower() if region is not None else "global"
        outputsaver.save_figure(
            fig,
            diagnostic_product=diagnostic_product,
            extra_keys={"var": var, "region": region},
            metadata={"description": description},
            extension=save_format,
            dpi=dpi,
            rebuild=rebuild,
        )

    def _regrid(self, data, tgt_grid_name, regrid_method):
        """Regrid a map to the target grid, so maps of different models can be compared."""
        self.logger.info("Regridding %s to %s with method %s", data.name, tgt_grid_name, regrid_method)
        regridder = Regridder(data=data, loglevel=self.loglevel)
        regridder.weights(tgt_grid_name=tgt_grid_name, regrid_method=regrid_method)
        return regridder.regrid(data)

    def subregion_selection(
        self,
        data=None,
        model=None,
        exp=None,
        mask_northern_boundary=None,
        northern_boundary_latitude=None,
        mask_southern_boundary=None,
        southern_boundary_latitude=None,
        lon_lim=None,
        lat_lim=None,
        region_name=None,
    ):
        """
        Selecting sub-region based on lon-lat bounds and applying boundary masks.

        Args:
            data (xarray.DataArray, optional): The data to be subsetted.
            model (str, optional): The model string identifier. Masks are applied if 'ICON' is in the model name.
            exp (str, optional): Experiment name.
            mask_northern_boundary (bool, optional): Whether to apply masking at the northern boundary.
            northern_boundary_latitude (float, optional): The latitude defining the northern cutoff.
            mask_southern_boundary (bool, optional): Whether to apply masking at the southern boundary.
            southern_boundary_latitude (float, optional): The latitude defining the southern cutoff.
            lon_lim (list[float], optional): Longitude limits [min, max] for the subregion.
            lat_lim (list[float], optional): Latitude limits [min, max] for the subregion.
            region_name (str, optional): String identifier for the region. Used for logging.

        Returns:
            xarray.DataArray: The subregion-selected and potentially masked data.
        """
        self.logger.info(f"Selecting the sub-region plots: {region_name}.")
        # Apply masking if necessary
        is_icon = "ICON" in (model or "")
        if is_icon and mask_northern_boundary and northern_boundary_latitude:
            data = data.where(data.lat < northern_boundary_latitude)
        if is_icon and mask_southern_boundary and southern_boundary_latitude:
            data = data.where(data.lat > southern_boundary_latitude)

        area_sel = AreaSelection(loglevel=self.loglevel)

        return area_sel.select_area(data, lon=lon_lim, lat=lat_lim, drop=True)
