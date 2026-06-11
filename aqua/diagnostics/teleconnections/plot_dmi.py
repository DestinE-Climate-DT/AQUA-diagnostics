import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import indexes_plot
from aqua.core.logger import log_configure
from aqua.diagnostics.base import TitleBuilder

from .base import PlotBaseMixin


class PlotDMI(PlotBaseMixin):
    """
    Class for plotting the DMI index.
    This class inherits from the PlotBaseMixin and implements the necessary methods
    to plot the DMI index.
    """
    def __init__(
        self,
        indexes=None,
        ref_indexes=None,
        outputdir: str = './',
        rebuild: bool =True,
        loglevel: str = "WARNING",
    ):
        """
        Plot the DMI index.

        Args:
            indexes (list): List of DMI index objects to plot.
            ref_indexes (list): List of reference DMI index objects to plot.
            outputdir (str): Directory to save the plot. Default is './'.
            rebuild (bool): Whether to rebuild the plot if it already exists. Default is True.
            loglevel (str): Logging level. Default is 'WARNING'.
        """
        super().__init__(
            indexes=indexes,
            ref_indexes=ref_indexes,
            diagnostic='dmi',
            outputdir=outputdir,
            rebuild=rebuild,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_name="PlotDMI", log_level=loglevel)

    def plot_index(self, thresh: float = 0.5, labels: list = None):
        """
        Plot the DMI index.

        Args:
            thresh (float): Threshold for highlighting significant values. Default is 0.5.
            labels (list): List of labels for the plot. Default is None.
        """
        # Join the indexes in a single list
        indexes = self.indexes + self.ref_indexes

        labels = super().set_labels() if labels is None else labels

        title = TitleBuilder(
            diagnostic="DMI index", model=self.models, exp=self.exps, ref_model=self.ref_models, ref_exp=self.ref_exps
        ).generate()

        fig, axs = indexes_plot(
            indexes=indexes, thresh=thresh, suptitle=title, ylabel="DMI index", labels=labels, loglevel=self.loglevel
        )

        if isinstance(axs, plt.Axes):
            axs = [axs]

        return fig, axs

    def set_index_description(self):
        return super().set_index_description(index_name="DMI index")