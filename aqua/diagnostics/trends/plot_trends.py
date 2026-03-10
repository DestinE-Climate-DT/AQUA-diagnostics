import xarray as xr
from aqua.core.graphics import plot_single_map
from aqua.core.logger import log_configure
from aqua.core.util import to_list, get_realizations
from aqua.diagnostics.base import OutputSaver, TitleBuilder


class PlotTrends():

    def __init__(self,
                trend,
                diagnostic: str ='Trends',
                save_pdf: bool = True,
                save_png: bool = True,
                dpi: int = 300,
                outdir: str = './',
                cmap: str = 'RdBu_r',
                loglevel: str = 'WARNING'):
        """
        Class to plot trend maps.

        Args:
            trend (xr.DataArray): The trend data to plot.
            diagnostic (str): Name of the diagnostic to plot. Default is 'trends'.
            save_pdf (bool): Whether to save the plot as a PDF file. Default is True.
            save_png (bool): Whether to save the plot as a PNG file. Default is True.
            dpi (int): Dots per inch for the saved figure. Default is 300.
            outdir (str): Directory to save the output files. Default is './'.
            cmap (str): Colormap to use for the plot. Default is 'RdBu_r'.
            loglevel (str): Logging level. Default is 'WARNING'.
        """
        if isinstance(trend, xr.Dataset):
            self.short_name = list(trend.data_vars.keys())[0]
            self.data = trend[self.short_name]
        else:
            self.data = trend
        self.diagnostic = diagnostic
        self.save_pdf = save_pdf
        self.save_png = save_png
        self.dpi = dpi
        self.outdir = outdir
        self.cmap = cmap
        self.loglevel = loglevel

        self.logger = log_configure(log_level=self.loglevel, log_name=f'Plot{self.diagnostic.capitalize()}')

    def get_data_info(self):
        """
        We extract the data needed for labels, description etc
        from the data arrays attributes.

        The attributes are:
        - AQUA_catalog
        - AQUA_model
        - AQUA_exp
        - AQUA_region
        - startdate
        - enddate
        - short_name
        - long_name
        - units
        """
        self.catalog = self.data.attrs.get('AQUA_catalog', 'unknown_catalog')
        self.model = self.data.attrs.get('AQUA_model', 'unknown_model')
        self.exp = self.data.attrs.get('AQUA_exp', 'unknown_exp')
        self.region = self.data.attrs.get('AQUA_region', 'global')
        self.startdate = self.data.attrs.get('startdate', 'unknown_startdate')
        self.enddate = self.data.attrs.get('enddate', 'unknown_enddate')
        self.short_name = self.data.attrs.get('short_name', self.short_name)
        self.long_name = self.data.attrs.get('long_name', 'unknown_long_name')
        self.units = self.data.attrs.get('units', 'unknown_units')
        self.start_year = self.startdate[:4] if self.startdate != 'unknown_startdate' else None
        self.end_year = self.enddate[:4] if self.enddate != 'unknown_enddate' else None
        self.realization = get_realizations(self.data)

    def set_title(self):
        """
        We build the title for the plot using the TitleBuilder class.
        """
        self.title = TitleBuilder(
            diagnostic=self.diagnostic,
            variable=self.long_name,
            regions=to_list(self.region) if self.region != 'global' else None,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            startyear=self.start_year,
            endyear=self.end_year).generate()

    def set_description(self):
        """
        We build the description for the plot using the TitleBuilder class.
        """
        description = (
            f"Trend map of {self.long_name} ({self.short_name}) for the period {self.startdate} to {self.enddate} "
            f"from the {self.catalog} catalog, model {self.model}, experiment {self.exp}."
        )

        return description

    def save_figure(self, fig, description: str):
        """
        Save the figure using the OutputSaver class.

        Args:
            fig: The figure to save.
            description: The description to save in the metadata.
        """
        outputsaver = OutputSaver(
            diagnostic=self.diagnostic,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            realization=self.realization,
            outputdir=self.outdir,
            loglevel=self.loglevel
        )

        metadata = {"Description": description}
        extra_keys = {}
        extra_keys.update({'var': self.short_name})
        if self.region != 'global':
            extra_keys.update({'region': self.region})

        outputsaver.save_figure(
            fig,
            diagnostic_product=self.diagnostic,
            metadata=metadata,
            extra_keys=extra_keys,
            save_pdf=self.save_pdf,
            save_png=self.save_png,
            dpi=self.dpi
        )

    def plot_trend(self, vmin: float = None, vmax: float = None):
        """
        We plot the trend map using the plot_single_map function.
        """
        self.get_data_info()
        self.set_title()
        description = self.set_description()

        fig, ax = plot_single_map(
            data=self.data,
            title=self.title,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            sym=False if vmin is not None and vmax is not None else True,
            return_fig=True,
        )

        self.save_figure(fig, description)
