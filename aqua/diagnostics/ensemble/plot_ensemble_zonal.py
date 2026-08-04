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
    ):
        """
        Plot ensemble mean and standard deviation of zonal averages in Lev-Lat coordinates.

        This method generates contour plots of the ensemble mean and standard deviation
        for a given variable on a latitude vs. vertical level (Lev) grid. The resulting
        plots can be saved as PNG and/or PDF files using the `save_figure` method.

        Args:
            var (str): Name of the variable to plot.
            dataset_mean (xarray.DataArray or xarray.Dataset): Ensemble mean data.
            dataset_std (xarray.DataArray or xarray.Dataset): Ensemble standard deviation data.
            description (str, optional): Description for saving the plots.
            title_mean (str, optional): Title for the mean plot. Auto-generated if None.
            title_std (str, optional): Title for the standard deviation plot. Auto-generated if None.
            figure_size (list[int], optional): Figure size [width, height]. Default is [10, 8].
            cbar_label (str, optional): Label for the colorbar.
            save_format (str or list, optional): Format(s) to save plots in (e.g. 'png', 'pdf', 'svg'). Default is SAVE_FORMAT.
            dpi (int, optional): Resolution for saved figures. Default is 300.
            units (str, optional): Units of the variable. Used in titles and labels if provided.
            ylim (tuple, optional): Y-axis limits for the plot (vertical levels). Default is (5500, 0).
            countour_levels (int, optional): Number of contour levels. Default is 20.
            cmap (str, optional): Colormap to use. Default is "RdBu_r".
            ylabel (str, optional): Label for y-axis. Default is "Depth (in m)".
            xlabel (str, optional): Label for x-axis. Default is "Latitude (in deg North)".
            data_name (str, optional): in order to safe plots with different names. 

        Returns:
            dict: Dictionary containing figure and axes objects for mean and std plots::

                {"mean_plot": [fig1, ax1], "std_plot": [fig2, ax2]}

        Raises:
            NoDataError: If `dataset_mean` or `dataset_std` is None.

        Notes:
            - Automatically generates titles for mean and STD if not provided.
            - Uses `self.save_figure` to save the plots as PNG and PDF.
            - Designed for zonal mean visualizations in Lev-Lat coordinates.
            - Default y-axis (vertical levels) is set to descend from 5500 m to 0 m.

        TODO:
            - Add support for multiple variables in a single call.
            - Include optional overlay of observations or reference zonal datasets.
            - Improve automatic scaling of colorbars for multiple variables or ensembles.
            - Add interactive plotting options.
        """
        self.logger.info("Plotting the ensemble computation of ensemble {data_name} zonal-averages for variable {self.var}")

        if title is None:
            title = TitleBuilder(diagnostic=description, model=self.model).generate()
        if (dataset is None):
            self.logger.warning(f"Ensemble Zonal data not provided for plotting. Skipping plotting!")        
            return None

        if isinstance(dataset, xr.Dataset):
            dataset = dataset[var]
        self.logger.info("Plotting ensemble Zonal-average")

        # Define the candidate keys in order of preference
        _coords = CoordIdentifier(dataset.coords)
        coords = _coords.identify_coords

        for k in VERTICAL_CANDIDATES:
            if coords.get(k) is not None:
                vert_coord = coords[k]["name"]

        # return if no vertical coordinate is found
        if vert_coord is None:
            self.logger.warning("No vertical coordinate found in Zonal data for {var}. Skipping it!")
            return 

        # do the selection on the first vertical coordinate found
        if len(vert_coord) > 1:
            self.logger.warning("Skipping plotting due to more than one vertical coordinate : %s", vert_coord)
            return 

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
        ax.set_title(title_mean)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, extend="both")
        cbar.set_label(cbar_label)
        self.logger.info("Saving Lev-Lon Zonal-average ensemble-mean as pdf and png")

        # Saving plots
        self.save_figure(var=var, fig=fig, data_name=data_name, description=description, format=save_format, dpi=dpi)

        return fig, ax
