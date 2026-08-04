import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from aqua.core.exceptions import NoDataError
from aqua.core.graphics import plot_single_map, plot_single_map_diff
from aqua.core.util import get_projection
from aqua.diagnostics.base import SAVE_FORMAT, TitleBuilder

from .base import BaseMixin

xr.set_options(keep_attrs=True)

class PlotEnsembleLatLon(BaseMixin):
    """Class to plot the ensmeble lat-lon"""

    # TODO: support sub-region selection and reggriding option

    def __init__(
        self,
        diagnostic_product: str = "EnsembleLatLon",
        catalog_list: list[str] = None,
        model_list: list[str] = None,
        exp_list: list[str] = None,
        source_list: list[str] = None,
        ref_catalog: str = None,
        ref_model: str = None,
        ref_exp: str = None,
        region: str = None,
        outputdir="./",
        loglevel: str = "WARNING",
    ):
        """
        Class for plotting ensemble latitude-longitude (Lat-Lon) data.

        This class inherits from `BaseMixin` and provides functionality to generate
        plots of ensemble datasets on a latitude-longitude grid. It supports
        multiple catalogs, models, experiments, and sources, and allows saving
        plots as PNG or PDF files. The class is intended for ensemble statistics
        visualization, such as mean and standard deviation maps.

        Args:
            diagnostic_product (str, optional): Name of the diagnostic product.
                Defaults to "EnsembleLatLon".
            catalog_list (list[str], optional): List of catalog names. If None, assigned to 'None_catalog'.
            model_list (list[str], optional): List of model names. If None, assigned to 'None_model'.
            exp_list (list[str], optional): List of experiment names. If None, assigned to 'None_exp'.
            source_list (list[str], optional): List of data source names. If None, assigned to 'None_source'.
            region (str, optional): Name of the region for plotting. Defaults to None.
            outputdir (str, optional): Directory to save output plots. Defaults to "./".
            loglevel (str, optional): Logging level. Defaults to "WARNING".

        Attributes:
            figure (matplotlib.figure.Figure or None): The figure object for the plot.
            diagnostic_product (str): Name of the diagnostic product being visualized.
            catalog_list (list[str]): List of catalogs being processed.
            model_list (list[str]): List of models being processed.
            exp_list (list[str]): List of experiments being processed.
            source_list (list[str]): List of sources being processed.
            region (str): Region name for plotting.
            outputdir (str): Directory path for saving plots.
            loglevel (str): Logging level for messages.

        Notes:
            - Designed to visualize ensemble mean and standard deviation on Lat-Lon grids.
            - Integrates with `BaseMixin` for consistent handling of catalogs, models, and experiments.
            - Uses `self.save_figure` for saving output plots in PNG and PDF formats.

        TODO:
            - Support sub-region selection for plotting.
            - Add regridding option for datasets with different grids.
            - Include automatic handling of color scales and legends for multiple ensemble members.
            - Add methods to overlay observations or reference datasets.
            - Enable interactive plotting for enhanced analysis.
        """
        self.diagnostic_product = diagnostic_product
        self.catalog_list = catalog_list
        self.model_list = model_list
        self.exp_list = exp_list
        self.source_list = source_list
        self.ref_catalog = ref_catalog
        self.ref_model = ref_model
        self.ref_exp = ref_exp

        self.region = region

        self.outputdir = outputdir
        self.loglevel = loglevel

        super().__init__(
            loglevel=self.loglevel,
            diagnostic_product=self.diagnostic_product,
            catalog_list=self.catalog_list,
            model_list=self.model_list,
            exp_list=self.exp_list,
            source_list=self.source_list,
            ref_catalog=self.ref_catalog,
            ref_model=self.ref_model,
            ref_exp=self.ref_exp,
            outputdir=self.outputdir,
        )

    def plot(
        self,
        var: str = None,
        dataset=None,
        long_name=None,
        description=None,
        dpi=300,
        title=None,
        save_format=SAVE_FORMAT,
        vmin=None,
        vmax=None,
        proj="robinson",
        proj_params={},
        transform_first=False,
        cyclic_lon=True,
        contour=True,
        coastlines=True,
        cbar_label=None,
        units=None,
        cmap=None,
        data_name=None,
    ):
        """
        Plot ensemble mean and standard deviation on a latitude-longitude map.

        Generates 2D maps of ensemble mean and standard deviation for a given
        variable using the specified projection and visualization options.
        The resulting figures can be saved as PNG and/or PDF files.

        Args:
            var (str): Variable name to plot.
            dataset_mean (xarray.DataArray or Dataset): Ensemble mean dataset.
            dataset_std (xarray.DataArray or Dataset): Ensemble standard deviation dataset.
            long_name (str, optional): Long descriptive name for the variable. Defaults to None.
            description (str, optional): Description string for saving the plot. Defaults to None.
            dpi (int, optional): Resolution for saved figures. Default is 300.
            title_mean (str, optional): Title for mean plot. Auto-generated if None.
            title_std (str, optional): Title for standard deviation plot. Auto-generated if None.
            save_format (str or list, optional): Format(s) to save figures in (e.g. 'png', 'pdf', 'svg').
                Default is SAVE_FORMAT.
            vmin_mean, vmax_mean (float, optional): Color scale limits for mean plot. Auto-set if None.
            vmin_std, vmax_std (float, optional): Color scale limits for std plot. Auto-set if None.
            proj (str, optional): Map projection. Default is "robinson".
            proj_params (dict, optional): Extra parameters for the projection. Defaults to {}.
            transform_first (bool, optional): Whether to transform data before plotting. Default is False.
            cyclic_lon (bool, optional): Whether longitude is cyclic. Default is False.
            contour (bool, optional): Overlay contours. Default is True.
            coastlines (bool, optional): Draw coastlines. Default is True.
            cbar_label (str, optional): Label for the colorbar. Auto-generated if None.
            units (str, optional): Units of the variable. Used for titles and labels.

        Returns:
            dict: Dictionary containing figure and axes for mean and std plots:
                  {'mean_plot': [fig1, ax1], 'std_plot': [fig2, ax2]}.
                  If standard deviation is zero everywhere, only 'mean_plot' is returned.

        Raises:
            NoDataError: If `dataset_mean` or `dataset_std` is None.

        Notes:
            - Titles and colorbar labels are automatically generated if not provided.
            - Uses `self.save_figure` to save figures in the formats specified.
            - Handles both xarray.DataArray and Dataset inputs.
            - If vmin_std equals vmax_std, std plot is skipped.

        TODO:
            - Add support for plotting multiple variables in one call.
            - Overlay observational or reference datasets.
            - Enable interactive plotting with cartopy or matplotlib widgets.
            - Improve handling of cyclic longitude for global datasets.
        """
        self.logger.info("Plotting the ensemble computation")
        if (dataset is None):
            self.logger.warning("No data given to the ensemble the plotting function")
            return 
        
        # Load the data into the memory
        dataset.load()

        if (isinstance(dataset, xr.Dataset)):
            dataset = dataset[var]
        if bool(dataset.isnull().all()) or bool((dataset == 0).all()):
            self.logger.warning(f"The map is empty (all NaN or all zero. Skipping the ensemble maps for {var} and {data_name}")
            return

        if cbar_label is None:
            cbar_label = var

        if long_name is None:
            long_name = dataset.attrs.get("long_name") or var

        if title is None:
            title = TitleBuilder(diagnostic="Ensemble diagnostic", variable=long_name, model=self.model).generate()

        proj = get_projection(proj, **proj_params)

        # mean plot
        if dataset is not None:
            fig, ax = plot_single_map(
                data=dataset,
                proj=proj,
                proj_params=proj_params,
                contour=contour,
                cyclic_lon=cyclic_lon,
                coastlines=coastlines,
                # transform_first=transform_first,
                return_fig=True,
                title=title,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                loglevel=self.loglevel,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            self.logger.debug("Saving 2D ensemble map")
            self.save_figure(var=var, fig=fig, data_name=data_name, description=description, format=save_format, dpi=dpi)
            return fig, ax
        else:
            return

    def plot_ensemble_diff_bias(
        self,
        var: str = None,
        dataset=None,
        ref_dataset=None,
        long_name=None,
        description=None,
        dpi=300,
        title=None,
        save_format=SAVE_FORMAT,
        vmin=None,
        vmax=None,
        proj="robinson",
        proj_params={},
        transform_first=False,
        cyclic_lon=True,
        contour=True,
        coastlines=True,
        cbar_label=None,
        units=None,
        cmap=None,
        data_name=None,
    ):
        """
        Plot ensemble mean and standard deviation on a latitude-longitude map.

        Generates 2D maps of ensemble mean and standard deviation for a given
        variable using the specified projection and visualization options.
        The resulting figures can be saved as PNG and/or PDF files.

        Args:
            var (str): Variable name to plot.
            dataset_mean (xarray.DataArray or Dataset): Ensemble mean dataset.
            dataset_std (xarray.DataArray or Dataset): Ensemble standard deviation dataset.
            long_name (str, optional): Long descriptive name for the variable. Defaults to None.
            description (str, optional): Description string for saving the plot. Defaults to None.
            dpi (int, optional): Resolution for saved figures. Default is 300.
            title_mean (str, optional): Title for mean plot. Auto-generated if None.
            title_std (str, optional): Title for standard deviation plot. Auto-generated if None.
            save_format (str or list, optional): Format(s) to save figures in (e.g. 'png', 'pdf', 'svg').
                Default is SAVE_FORMAT.
            vmin_mean, vmax_mean (float, optional): Color scale limits for mean plot. Auto-set if None.
            vmin_std, vmax_std (float, optional): Color scale limits for std plot. Auto-set if None.
            proj (str, optional): Map projection. Default is "robinson".
            proj_params (dict, optional): Extra parameters for the projection. Defaults to {}.
            transform_first (bool, optional): Whether to transform data before plotting. Default is False.
            cyclic_lon (bool, optional): Whether longitude is cyclic. Default is False.
            contour (bool, optional): Overlay contours. Default is True.
            coastlines (bool, optional): Draw coastlines. Default is True.
            cbar_label (str, optional): Label for the colorbar. Auto-generated if None.
            units (str, optional): Units of the variable. Used for titles and labels.
            data_name (str, optional): in order to safe plots with different names.
        Returns:
            dict: Dictionary containing figure and axes for mean and std plots:
                  {'mean_plot': [fig1, ax1], 'std_plot': [fig2, ax2]}.
                  If standard deviation is zero everywhere, only 'mean_plot' is returned.

        Raises:
            NoDataError: If `dataset_mean` or `dataset_std` is None.

        Notes:
            - Titles and colorbar labels are automatically generated if not provided.
            - Uses `self.save_figure` to save figures in the formats specified.
            - Handles both xarray.DataArray and Dataset inputs.
            - If vmin_std equals vmax_std, std plot is skipped.

        TODO:
            - Add support for plotting multiple variables in one call.
            - Overlay observational or reference datasets.
            - Enable interactive plotting with cartopy or matplotlib widgets.
            - Improve handling of cyclic longitude for global datasets.
        """
        self.logger.info("Plotting the ensemble computation")
        if (dataset is not None and ref_dataset is not None):
            self.logger.debug("Data given to the ensemble bias the plotting function")
        else:
            self.logger.warning("No data given to the ensemble bias the plotting function. Skipping plotting maps!")
            return 

        # Load the data into the memory
        dataset.load()
        ref_dataset.load()

        if (isinstance(dataset, xr.Dataset)):
            dataset = dataset[var]
        if bool(dataset.isnull().all()) or bool((dataset == 0).all()):
            self.logger.warning(f"The map is empty (all NaN or all zero. Skipping the ensemble maps for {var} and {data_name}")
            return

        if cbar_label is None:
            cbar_label = var

        if long_name is None:
            long_name = dataset.attrs.get("long_name") or var

        if title is None:
            title = TitleBuilder(diagnostic="Bias ensemble map", variable=long_name, model=self.model).generate()

        proj = get_projection(proj, **proj_params)

        if isinstance(dataset, xr.Dataset):
            dataset = dataset[var]
        if isinstance(ref_dataset, xr.Dataset):
            ref_dataset = ref_dataset[var]

        # bias plot
        fig, ax = plot_single_map_diff(
            data=dataset,
            data_ref=ref_dataset,
            proj=proj,
            proj_params=proj_params,
            contour=contour,
            cyclic_lon=cyclic_lon,
            coastlines=coastlines,
            # transform_first=transform_first,
            return_fig=True,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            loglevel=self.loglevel,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        self.logger.debug("Saving 2D ensemble bias map")
        self.save_figure(var=var, fig=fig, data_name=data_name, description=description, format=save_format, dpi=dpi)
        return fig, ax            

