import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.base import SAVE_FORMAT, TitleBuilder
from aqua.core.data_model import CoordIdentifier

from .base import BaseMixin

xr.set_options(keep_attrs=True)

VERTICAL_CANDIDATES = ('isobaric', 'depth', 'height')

class PlotEnsembleZonal(BaseMixin):
    def __init__(
        self,
        diagnostic_product: str = "EnsembleZonal",
        catalog_list: list[str] = None,
        model_list: list[str] = None,
        exp_list: list[str] = None,
        source_list: list[str] = None,
        region: str = None,
        outputdir="./",
        loglevel: str = "WARNING",
    ):
        """
        Class for plotting ensemble zonal mean data.

        This class inherits from `BaseMixin` and provides functionality to
        visualize ensemble datasets as zonal averages. It supports multiple
        catalogs, models, experiments, and sources, and allows specifying a
        region for the analysis. The resulting plots can be saved to a
        specified output directory.

        Args:
            diagnostic_product (str, optional): Name of the diagnostic product.
                Defaults to "EnsembleZonal".
            catalog_list (list[str], optional): List of catalog names. If None,
                assigned to 'None_catalog'.
            model_list (list[str], optional): List of model names. If None,
                assigned to 'None_model'.
            exp_list (list[str], optional): List of experiment names. If None,
                assigned to 'None_exp'.
            source_list (list[str], optional): List of source names. If None,
                assigned to 'None_source'.
            region (str, optional): Name of the region for zonal averaging. Defaults to None.
            outputdir (str, optional): Directory path to save plots. Defaults to "./".
            loglevel (str, optional): Logging level. Defaults to "WARNING".

        Attributes:
            diagnostic_product (str): Name of the diagnostic product.
            catalog_list (list[str]): List of catalogs being processed.
            model_list (list[str]): List of models being processed.
            exp_list (list[str]): List of experiments being processed.
            source_list (list[str]): List of sources being processed.
            region (str): Region used for zonal analysis.
            outputdir (str): Output directory for saving plots.
            loglevel (str): Logging level for messages.

        TODO:
            - Add support for sub-region selection.
            - Add optional regridding of input datasets.
            - Include automatic color scale adjustment for multi-model ensembles.
            - Add functionality to overlay observational or reference zonal datasets.
        """
        self.diagnostic_product = diagnostic_product
        self.catalog_list = catalog_list
        self.model_list = model_list
        self.exp_list = exp_list
        self.source_list = source_list
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
            outputdir=self.outputdir,
        )

    def plot(
        self,
        var: str = None, 
        dataset=None,
        data_name=None,
        description=None,
        title=None,
        figure_size=[10, 8],
        cbar_label=None,
        save_format=SAVE_FORMAT,
        dpi=300,
        units=None,
        ylim=(5500, 0),
        countour_levels=20,
        cmap="RdBu_r",
        ylabel="Depth (in m)",
        xlabel="Latitude (in deg North)",
        rebuild=True,
    ):
        """
        Plot zonal averages in Level-Latitude coordinates.

        This method generates contour plots for a given variable on a latitude 
        vs. vertical level (Lev) grid. The resulting plots can be automatically 
        saved as PNG and/or PDF files.

        Args:
            var (str, optional): Name of the variable to plot. Defaults to None.
            dataset (xarray.DataArray or xarray.Dataset, optional): The 2D zonal dataset 
                (vertical level vs. latitude) to be plotted. Defaults to None.
            data_name (str, optional): File naming label to distinguish saved plots 
                (e.g., 'mean', 'std'). Defaults to None.
            description (str, optional): Description used for auto-generating the title 
                and saving the plots. Defaults to None.
            title (str, optional): Title for the plot. Auto-generated if None. Defaults to None.
            figure_size (list[int], optional): Figure size [width, height]. Defaults to [10, 8].
            cbar_label (str, optional): Label for the colorbar. Defaults to None.
            save_format (str or list, optional): Format(s) to save plots in (e.g., 'png', 'pdf'). 
                Defaults to SAVE_FORMAT.
            dpi (int, optional): Resolution for saved figures. Defaults to 300.
            units (str, optional): Units of the variable. Defaults to None.
            ylim (tuple, optional): Y-axis limits for the plot (vertical levels). 
                Defaults to (5500, 0) for descending depth.
            countour_levels (int, optional): Number of contour levels to plot. Defaults to 20.
            cmap (str, optional): Colormap to use. Defaults to "RdBu_r".
            ylabel (str, optional): Label for y-axis. Defaults to "Depth (in m)".
            xlabel (str, optional): Label for x-axis. Defaults to "Latitude (in deg North)".
            rebuild (bool, optional): Whether to rebuild the output file path in `save_figure`. Defaults to True.

        Returns:
            tuple or None: A tuple containing the `(matplotlib.figure.Figure, matplotlib.axes.Axes)` 
            objects if plotting succeeds, or `None` if no dataset is provided or plotting fails due to 
            missing vertical coordinates.

        Notes:
            - Automatically detects vertical coordinates (isobaric, depth, height).
            - Automatically generates titles if not provided.
            - Uses `self.save_figure` to save the plots.
            - Dimensions outside of vertical and latitude are squeezed.
        """
        self.logger.info("Plotting the ensemble computation of ensemble {data_name} zonal-averages for variable {self.var}")

        if title is None:
            title = TitleBuilder(diagnostic=description, model=self.model).generate()
        if (dataset is None):
            self.logger.warning(f"Ensemble Zonal data not provided for plotting. Skipping plotting!")        
            return None

        if isinstance(dataset, xr.Dataset):
            dataset = dataset[var]
        self.logger.info(f"Plotting ensemble Zonal-average for {var}")

        # Define the candidate keys in order of preference
        _coords = CoordIdentifier(dataset.coords)
        coords = _coords.identify_coords()

        for k in VERTICAL_CANDIDATES:
            if coords.get(k) is not None:
                vert_coord = coords[k]["name"]
                self.logger.info(f"Vertical coordiante {vert_coord} for {var}")

        # check if a vertical coordinate is found
        if len(vert_coord) is None:
            self.logger.warning(f"Skipping plotting due to missing vertical coordinate for {var}")
            return 

        ## do the selection on the first vertical coordinate found
        #if len(vert_coord) > 1:
        #    self.logger.warning("Skipping plotting due to more than one vertical coordinate : %s", vert_coord)
        #    return 

        # squeeze all other dimensions if present
        dataset = dataset.squeeze()

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.contourf(
            dataset["lat"],  # Safely hardcoded to "lat"
            dataset[vert_coord],
            dataset,
            cmap=cmap,
            levels=countour_levels,
            extend="both",
        )
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_facecolor("grey")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, extend="both")
        cbar.set_label(cbar_label)
        self.logger.info("Saving Lev-Lon Zonal-average ensemble-mean as pdf and png")

        # Saving plots
        self.save_figure(var=var, fig=fig, data_name=data_name, description=description, format=save_format, rebuild=rebuild, dpi=dpi)

        return fig, ax
