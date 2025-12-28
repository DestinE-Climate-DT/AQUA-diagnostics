import matplotlib.pyplot as plt

from aqua.core.graphics import plot_histogram
from aqua.core.logger import log_configure
from aqua.core.util import to_list, DEFAULT_REALIZATION
from aqua.diagnostics.base import OutputSaver


class PlotHistogram():
    """
    Class for plotting Histogram diagnostics.
    Provides methods to plot histogram/PDF data with customizable labels,
    titles, and styling options.
    """
    def __init__(self, data=None, ref_data=None,
                 diagnostic_name='histogram',
                 density=True,
                 loglevel: str = 'WARNING'):
        """
        Initialize the PlotHistogram class.

        Args:
            data: List of histogram DataArrays to plot, or single DataArray.
            ref_data: Reference histogram DataArray.
            diagnostic_name (str): Name of the diagnostic. Default is 'histogram'.
            density (bool): Whether data represents PDF (True) or counts (False).
            loglevel (str): Logging level. Default is 'WARNING'.
        """
        self.loglevel = loglevel
        self.logger = log_configure(loglevel, 'PlotHistogram')

        self.data = to_list(data) if data is not None else []
        self.ref_data = ref_data
        self.diagnostic_name = diagnostic_name
        self.density = density

        self.len_data = len(self.data)
        self.len_ref = 1 if ref_data is not None else 0
        
        self.get_data_info()

    def get_data_info(self):
        """Extract metadata from data arrays."""
        self.catalogs, self.models, self.exps = [], [], []
        self.realizations = []
        self.region = None
        self.short_name = None
        self.standard_name = None
        self.long_name = None
        self.units = None
        
        # Extract metadata from data arrays
        for data_item in self.data:
            if data_item is not None and hasattr(data_item, 'AQUA_catalog'):
                self.catalogs.append(data_item.AQUA_catalog)
                self.models.append(data_item.AQUA_model)
                self.exps.append(data_item.AQUA_exp)

                # Extract realization if available
                if hasattr(data_item, 'AQUA_realization'):
                    self.realizations.append(data_item.AQUA_realization)
                    self.logger.debug(f'Extracted realization: {data_item.AQUA_realization}')
                else:
                    self.realizations.append(DEFAULT_REALIZATION)
                    self.logger.debug(f'No realization found in data, using default: {DEFAULT_REALIZATION}')

                # Extract region if not already set
                if self.region is None and hasattr(data_item, 'AQUA_region'):
                    self.region = data_item.AQUA_region

                # Extract variable names if not already set
                if self.short_name is None and hasattr(data_item, 'short_name'):
                    self.short_name = data_item.short_name
                if self.standard_name is None and hasattr(data_item, 'standard_name'):
                    self.standard_name = data_item.standard_name
                if self.long_name is None and hasattr(data_item, 'long_name'):
                    self.long_name = data_item.long_name
                if self.units is None and hasattr(data_item, 'center_of_bin'):
                    self.units = getattr(data_item.center_of_bin, 'units', None)
        
        self.logger.debug(f'Extracted metadata for {len(self.models)} datasets: {list(zip(self.models, self.exps))}')
        self.logger.debug(f'Extracted realizations: {self.realizations}')
        self.logger.debug(f'Extracted region: {self.region}')

    def set_data_labels(self):
        """Set the data labels for the plot."""
        data_labels = []
        num_labels = max(len(self.models), len(self.exps), 1)
        
        for i in range(num_labels):
            if i < len(self.models) and i < len(self.exps):
                data_labels.append(f'{self.models[i]} {self.exps[i]}')
            else:
                data_labels.append(f'Dataset {i+1}')
        
        self.logger.debug('Data labels: %s', data_labels)
        return data_labels
    
    def set_ref_label(self):
        """Set the reference label for the plot."""
        ref_label = None
        
        if self.ref_data is not None:
            model = self.ref_data.attrs.get('AQUA_model', 'Unknown')
            exp = self.ref_data.attrs.get('AQUA_exp', 'Unknown')
            ref_label = f'{model} {exp}'
        
        self.logger.debug('Reference label: %s', ref_label)
        return ref_label

    def set_title(self):
        """Set the title for the plot."""
        # Use long_name as-is (already contains "Pdf of" or "Histogram of" from aqua.core.histogram)
        title = self.long_name or self.standard_name or self.short_name or ""
        
        if title:
            title += ' '
        
        if self.region is not None:
            title += f'[{self.region}] '

        if self.len_data == 1:
            title += f'for {self.catalogs[0]} {self.models[0]} {self.exps[0]} '

        self.logger.debug('Title: %s', title)
        return title

    def set_description(self):
        """Set the description for the plot."""
        description = ""
        
        # Start with PDF/Histogram type
        if self.density:
            description += "Probability density function (PDF) of "
        else:
            description += "Histogram of "
        
        # Variable name (long_name preferred)
        for name in [self.long_name, self.standard_name, self.short_name]:
            if name is not None:
                # Remove "Pdf of" or "Histogram of" prefix if present
                name_clean = name.replace("Pdf of ", "").replace("Histogram of ", "")
                description += f"{name_clean} "
                break
        
        # Units
        if self.units is not None:
            description += f"[{self.units}] "
        
        # Short name in parentheses (if different from what was already used)
        if self.short_name is not None and self.long_name is not None:
            description += f"({self.short_name}) "
        
        # Region - only if not Global
        if self.region is not None and self.region.lower() != 'global':
            description += f"over {self.region} "
        
        # Extract dates from data
        data_item = self.data[0] if self.data else None
        ref_item = self.ref_data
        
        data_pair = (getattr(data_item, 'AQUA_startdate', None), 
                    getattr(data_item, 'AQUA_enddate', None))
        ref_pair = (getattr(ref_item, 'AQUA_startdate', None),
                    getattr(ref_item, 'AQUA_enddate', None))
        
        # Smart date display: show dates only once if they are the same
        if data_pair == ref_pair and data_pair != (None, None):
            # Same period for model and reference
            description += f"for {self.models[0]}/{self.exps[0]}"
            if ref_item is not None:
                ref_model = getattr(ref_item, 'AQUA_model', 'reference')
                description += f" vs {ref_model}"
            description += f" from {data_pair[0]} to {data_pair[1]}"
        else:
            # Different periods
            if data_pair != (None, None):
                description += f"for {self.models[0]}/{self.exps[0]} "
                description += f"from {data_pair[0]} to {data_pair[1]}"
            if ref_pair != (None, None):
                ref_model = getattr(ref_item, 'AQUA_model', 'reference')
                description += f", {ref_model} from {ref_pair[0]} to {ref_pair[1]}"
        
        description += '.'
        
        self.logger.warning('Description: %s', description)
        return description

    def plot(self, data_labels=None, ref_label=None, title=None, 
             style=None, xlogscale=False, ylogscale=True,
             xmax=None, xmin=None, ymax=None, ymin=None,
             smooth=False, smooth_window=5):
        """
        Plot histogram data.
        
        Args:
            data_labels (list, optional): Labels for the data.
            ref_label (str, optional): Label for the reference data.
            title (str, optional): Title for the plot.
            style (str, optional): Plotting style.
            xlogscale (bool): Use log scale for x-axis.
            ylogscale (bool): Use log scale for y-axis.
            xmax (float, optional): Maximum x value.
            xmin (float, optional): Minimum x value.
            ymax (float, optional): Maximum y value.
            ymin (float, optional): Minimum y value.
            smooth (bool): Apply smoothing to data.
            smooth_window (int): Window size for smoothing.

        Returns:
            tuple: Matplotlib figure and axes objects.
        """
        return plot_histogram(
            data=self.data,
            ref_data=self.ref_data,
            data_labels=data_labels,
            ref_label=ref_label,
            title=title,
            style=style,
            xlogscale=xlogscale,
            ylogscale=ylogscale,
            xmax=xmax,
            xmin=xmin,
            ymax=ymax,
            ymin=ymin,
            smooth=smooth,
            smooth_window=smooth_window,
            loglevel=self.loglevel
        )
    
    def save_plot(self, fig, 
                  description: str = None, 
                  rebuild: bool = True,
                  outputdir: str = './', 
                  dpi: int = 300, 
                  format: str = 'png'):
        """
        Save the plot to a file.

        Args:
            fig (matplotlib.figure.Figure): Figure object.
            description (str): Description of the plot.
            rebuild (bool): If True, rebuild the plot even if it already exists.
            outputdir (str): Output directory to save the plot.
            dpi (int): Dots per inch for the plot.
            format (str): Format of the plot ('png' or 'pdf').
        """
        metadata = {
            'catalog': getattr(self, 'catalogs', ['unknown_catalog'])[0],
            'model': getattr(self, 'models', ['unknown_model'])[0], 
            'exp': getattr(self, 'exps', ['unknown_exp'])[0]
        }
        
        # Add realization
        if self.realizations:
            metadata['realization'] = self.realizations[0]
            self.logger.debug(f'Using realization for plot filename: {self.realizations[0]}')

        # Use class attributes
        var = getattr(self, 'short_name', None) or getattr(self, 'standard_name', None)
        region = self.region

        extra_keys = {}
        if var: extra_keys['var'] = var
        if region: extra_keys['region'] = region.replace(' ', '').lower()
        
        outputsaver = OutputSaver(diagnostic=self.diagnostic_name, outputdir=outputdir,
                                  loglevel=self.loglevel, **metadata)
        
        diagnostic_product = f"{self.diagnostic_name}_pdf"
           
        if format == 'png':
            outputsaver.save_png(fig, diagnostic_product, extra_keys=extra_keys, 
                            metadata={'Description': description, 'dpi': dpi}, rebuild=rebuild)
        else:
            outputsaver.save_pdf(fig, diagnostic_product, extra_keys=extra_keys, 
                            metadata={'Description': description, 'dpi': dpi}, rebuild=rebuild)

    def run(self, outputdir='./', rebuild=True, dpi=300, style=None, 
            format='png', xlogscale=False, ylogscale=True,
            xmax=None, xmin=None, ymax=None, ymin=None,
            smooth=False, smooth_window=5, show=False):
        """
        Run the complete plotting workflow.
        
        Args:
            outputdir (str): Output directory to save the plot.
            rebuild (bool): If True, rebuild the plot even if it already exists.
            dpi (int): Dots per inch for the plot.
            style (str): Plotting style.
            format (str): Format of the plot ('png' or 'pdf').
            xlogscale (bool): Use log scale for x-axis.
            ylogscale (bool): Use log scale for y-axis.
            xmax (float, optional): Maximum x value.
            xmin (float, optional): Minimum x value.
            ymax (float, optional): Maximum y value.
            ymin (float, optional): Minimum y value.
            smooth (bool): Apply smoothing to data.
            smooth_window (int): Window size for smoothing.
            show (bool): If True, display the plot interactively.
        """
        self.logger.info('Running PlotHistogram')

        data_labels = self.set_data_labels()
        ref_label = self.set_ref_label()
        description = self.set_description()
        title = self.set_title()

        fig, _ = self.plot(data_labels=data_labels, ref_label=ref_label, 
                          title=title, style=style,
                          xlogscale=xlogscale, ylogscale=ylogscale,
                          xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin,
                          smooth=smooth, smooth_window=smooth_window)

        self.save_plot(fig, description=description, rebuild=rebuild,
                      outputdir=outputdir, dpi=dpi, format=format)
        
        if show:
            plt.show()
        plt.close(fig)
        
        self.logger.info('PlotHistogram completed successfully')