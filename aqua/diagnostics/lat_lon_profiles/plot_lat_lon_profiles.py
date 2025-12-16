import matplotlib.pyplot as plt

from aqua.core.graphics import plot_seasonal_lat_lon_profiles
from aqua.core.logger import log_configure
from aqua.core.util import to_list, strlist_to_phrase, DEFAULT_REALIZATION
from aqua.core.graphics import plot_lat_lon_profiles
from aqua.diagnostics.base import OutputSaver

class PlotLatLonProfiles():
    """
    Class for plotting Lat-Lon Profiles diagnostics.
    This class provides methods to set data labels, description, title,
    and to plot the data. It handles data arrays regardless of their original
    temporal frequency, as temporal averaging is handled upstream.
    """
    def __init__(self, data=None, ref_data=None,
                 data_type='longterm',
                 ref_std_data=None,
                 diagnostic_name='lat_lon_profiles',
                 loglevel: str = 'WARNING'):
        """
        Initialise the PlotLatLonProfiles class.
        This class is used to plot lat lon profiles data previously processed
        by the LatLonProfiles class.

        Args:
            data: Can be either:
                - List of temporally-averaged data arrays for annual plots
                - List of seasonal data [DJF, MAM, JJA, SON] for seasonal plots
            ref_data: Reference data (structure matches data based on data_type)
            data_type (str): 'longterm' for single/multi-line longterm plots, 'seasonal' for 4-panel seasonal plots
            ref_std_data: Reference standard deviation data
            diagnostic_name (str): Name of the diagnostic. Default is 'lat_lon_profiles'.
            loglevel (str): Logging level. Default is 'WARNING'.
            
        Note:
            data_type determines how 'data' is interpreted:
            - 'longterm': data should be list of DataArrays for single plot
            - 'seasonal': data should be [DJF, MAM, JJA, SON] for 4-panel seasonal plots
        """
        self.loglevel = loglevel
        self.logger = log_configure(loglevel, 'PlotLatLonProfiles')

        self.data_type = data_type

        # Store data based on type
        if data_type == 'longterm':
            self.data = to_list(data) if data is not None else []
            self.ref_data = ref_data
        elif data_type == 'seasonal':
            self.data = data if data is not None else []  # Store seasonal data directly in unified interface
            self.ref_data = ref_data  # Store seasonal ref data directly
        else:
            raise ValueError(f"data_type must be 'longterm' or 'seasonal', got '{data_type}'")

        self.ref_std_data = ref_std_data
        self.diagnostic_name = diagnostic_name

        self.len_data, self.len_ref = self._check_data_length()
        self.get_data_info()

    def set_data_labels(self):
        """Set the data labels for the plot based on data_type."""
        # Use self.models and self.exps to create labels
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
        """
        Set the reference label for the plot.
        The label is extracted from the reference data array attributes.

        Returns:
            ref_label (str): Reference label for the plot.
        """
        ref_label = None
        
        if self.ref_data is not None:
            # Handle seasonal (list) vs longterm (single DataArray)
            if self.data_type == 'seasonal' and isinstance(self.ref_data, list):
                ref_item = self.ref_data[0] if self.ref_data else None
            else:
                ref_item = self.ref_data 
            
            if ref_item is not None and hasattr(ref_item, 'AQUA_model'):
                model = ref_item.attrs.get('AQUA_model', 'Unknown')
                exp = ref_item.attrs.get('AQUA_exp', 'Unknown')
                ref_label = f'{model} {exp}'
        
        self.logger.debug('Reference label: %s', ref_label)
        return ref_label
    
    def get_data_info(self):
        """Extract metadata from data arrays based on data_type."""
        self.catalogs, self.models, self.exps = [], [], []
        self.realizations = []
        self.region = None
        self.short_name = None
        self.standard_name = None
        self.long_name = None
        self.units = None
        self.mean_type = None
        
        # Get all data items to extract metadata from
        data_items = []
        if self.data_type == 'longterm':
            data_items = self.data
        elif self.data_type == 'seasonal':
            # For seasonal, use first season's data
            first_season = self.data[0] if self.data else []
            data_items = first_season if isinstance(first_season, list) else [first_season]
        
        # Extract metadata from all data items
        for data_item in data_items:
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
                if self.units is None and hasattr(data_item, 'units'):
                    self.units = data_item.units

        # Set mean_type from first data item if not already set
        first_data = data_items[0] if data_items else None
        if first_data is not None and hasattr(first_data, 'AQUA_mean_type'):
            self.mean_type = first_data.AQUA_mean_type
        
        self.logger.debug(f'Extracted metadata for {len(self.models)} datasets: {list(zip(self.models, self.exps))}')
        self.logger.debug(f'Extracted realizations: {self.realizations}')
        self.logger.debug(f'Extracted region: {self.region}')
        
        # Handle std dates
        if self.ref_std_data is not None:
            self.std_startdate = getattr(self.ref_std_data, 'std_startdate', None)
            self.std_enddate = getattr(self.ref_std_data, 'std_enddate', None)
        else:
            self.std_startdate = None
            self.std_enddate = None

    def plot(self, data_labels=None, ref_label=None, title=None, style=None):
        """
        Unified plotting method that handles all plotting scenarios based on data_type.
        
        Args:
            data_labels (list, optional): Labels for the data.
            ref_label (str, optional): Label for the reference data.  
            title (str, optional): Title for the plot.
            style (str, optional): Plotting style. Default is the AQUA style.

        Returns:
            tuple: Matplotlib figure and axes objects.
        """
        if self.data_type == 'seasonal':
            # For seasonal plots, delegate to the specialized seasonal method
            return self.plot_seasonal_lines(data_labels=data_labels, title=title)
        
        data_to_plot = self.data
        ref_to_plot = self.ref_data
        
        # Call the graphics function
        return plot_lat_lon_profiles(
            data=data_to_plot,
            ref_data=ref_to_plot,
            ref_std_data=self.ref_std_data,
            data_labels=data_labels,
            ref_label=ref_label,
            title=title,
            style=style,
            loglevel=self.loglevel
        )
    
    def save_plot(self, fig, 
                  description: str = None, 
                  rebuild: bool = True,
                  outputdir: str = './', 
                  dpi: int = 300, 
                  format: str = 'png', 
                  diagnostic: str = None):
        """
        Save the plot to a file.

        Args:
            fig (matplotlib.figure.Figure): Figure object.
            description (str): Description of the plot.
            rebuild (bool): If True, rebuild the plot even if it already exists.
            outputdir (str): Output directory to save the plot.
            dpi (int): Dots per inch for the plot.
            format (str): Format of the plot ('png' or 'pdf'). Default is 'png'.
            diagnostic (str): Diagnostic name to be used in the filename as diagnostic_product.
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

        # Build extra_keys
        extra_keys = {}
        if var: extra_keys['var'] = var
        if region: extra_keys['region'] = region
        
        # diagnostic_product must match the one used in OutputSaver
        base_diagnostic = diagnostic if diagnostic else self.diagnostic_name
        outputsaver = OutputSaver(diagnostic=base_diagnostic, outputdir=outputdir,
                                  loglevel=self.loglevel, **metadata)
        
        # Build diagnostic_product with data_type info
        if self.data_type == 'seasonal':
            diagnostic_product = f"seasonal_{self.mean_type}_profile"
        else:  # longterm
            diagnostic_product = f"{self.mean_type}_profile"
        
        # Save based on format
        if format == 'png':
            outputsaver.save_png(fig, diagnostic_product, extra_keys=extra_keys, 
                            metadata={'description': description, 'dpi': dpi}, rebuild=rebuild)
        else:
            outputsaver.save_pdf(fig, diagnostic_product, extra_keys=extra_keys, 
                            metadata={'description': description, 'dpi': dpi}, rebuild=rebuild)

    def _check_data_length(self):
        """
        Check the length of the data arrays and reference data based on data_type.
        Returns:
            tuple: (length of data arrays, length of reference data)
        """
        len_data = len(self.data) if self.data else 0
        
        if self.data_type == 'longterm':
            len_ref = 1 if self.ref_data is not None else 0
        elif self.data_type == 'seasonal':
            len_ref = len(self.ref_data) if self.ref_data else 0
        else:
            len_ref = 0
        
        self.logger.debug(f'Data type: {self.data_type}, Data length: {len_data}, Reference length: {len_ref}')
        return len_data, len_ref

    def set_title(self):
        """
        Set the title for the plot.
        Specialized for Lat-Lon Profiles diagnostic.

        Returns:
            title (str): Title for the plot.
        """
        title = f"{self.mean_type.capitalize()} profile "


        for name in [self.long_name, self.standard_name, self.short_name]:
            if name is not None:
                title += f'for {name} '
                break
        if self.units is not None:
            title += f'[{self.units}] '

        if self.region is not None:
            title += f'[{self.region}] '

        if self.len_data == 1:
            title += f'for {self.catalogs[0]} {self.models[0]} {self.exps[0]} '

        self.logger.debug('Title: %s', title)
        return title

    def set_description(self):
        """
        Set the caption for the plot.

        Returns:
            description (str): Caption for the plot.
        """
        # Start with data_type info for seasonal plots
        if self.data_type == 'seasonal':
            description = "Seasonal "
        else:
            description = ""
        
        # Mean type (zonal/meridional) and variable name
        description += f"{self.mean_type} profile of "
        
        # Variable name
        for name in [self.long_name, self.standard_name, self.short_name]:
            if name is not None:
                description += f"{name} "
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

        # Dataset info with periods
        num_items = min(len(self.catalogs), len(self.models), len(self.exps)) if hasattr(self, 'catalogs') else 0
        
        # Collect all unique date pairs
        data_dates = []
        ref_dates = []
        std_dates = []
        
        if num_items > 0:
            description += "for "
            dataset_parts = []
            for i in range(num_items):
                catalog = self.catalogs[i] if i < len(self.catalogs) else 'unknown'
                model = self.models[i] if i < len(self.models) else 'unknown'
                exp = self.exps[i] if i < len(self.exps) else 'unknown'
                
                dataset_str = f"{catalog} {model} {exp}"
                
                # Extract dates from data
                data_items = self.data if self.data_type == 'longterm' else [self.data[0]] if self.data else []
                if i < len(data_items):
                    data_item = data_items[i]
                    data_start = getattr(data_item, 'AQUA_startdate', None)
                    data_end = getattr(data_item, 'AQUA_enddate', None)
                    if data_start and data_end:
                        data_dates.append((data_start, data_end))
                
                dataset_parts.append(dataset_str)
            
            description += strlist_to_phrase(dataset_parts)

        # Reference data description with period
        if self.len_ref > 0 and self.ref_data is not None:
            ref_item = self.ref_data if not isinstance(self.ref_data, list) else self.ref_data[0]
            
            ref_catalog = getattr(ref_item, 'AQUA_catalog', 'unknown')
            ref_model = getattr(ref_item, 'AQUA_model', 'unknown')
            ref_exp = getattr(ref_item, 'AQUA_exp', 'unknown')
            
            # Extract reference dates
            ref_start = getattr(ref_item, 'AQUA_startdate', None)
            ref_end = getattr(ref_item, 'AQUA_enddate', None)
            if ref_start and ref_end:
                ref_dates.append((ref_start, ref_end))
            
            description += f", compared to {ref_catalog} {ref_model} {ref_exp}"
        
        # Standard deviation info with period
        if self.ref_std_data is not None:
            std_start = getattr(self.ref_std_data, 'std_startdate', None)
            std_end = getattr(self.ref_std_data, 'std_enddate', None)
            if std_start and std_end:
                std_dates.append((std_start, std_end))
        
        # Smart date display logic
        # Check if we have all three date pairs
        has_data_dates = len(data_dates) > 0
        has_ref_dates = len(ref_dates) > 0
        has_std_dates = len(std_dates) > 0
        
        # Extract the actual date pairs (assume first entry if multiple)
        data_pair = data_dates[0] if has_data_dates else (None, None)
        ref_pair = ref_dates[0] if has_ref_dates else (None, None)
        std_pair = std_dates[0] if has_std_dates else (None, None)
        
        # Case 1: All three are identical
        if (has_data_dates and has_ref_dates and has_std_dates and 
            data_pair == ref_pair == std_pair):
            description += f" from {data_pair[0]} to {data_pair[1]} with ±2σ uncertainty bands"
        
        # Case 2: Data and reference are same, std is different (or missing)
        elif (has_data_dates and has_ref_dates and data_pair == ref_pair):
            description += f" from {data_pair[0]} to {data_pair[1]}"
            if has_std_dates and std_pair != data_pair:
                description += f" with ±2σ uncertainty bands computed over {std_pair[0]} to {std_pair[1]}"
            elif has_std_dates:
                description += " with ±2σ uncertainty bands"
        
        # Case 3: Reference and std are same, data is different
        elif (has_ref_dates and has_std_dates and ref_pair == std_pair):
            if has_data_dates:
                description += f" from {data_pair[0]} to {data_pair[1]}"
            description += f", reference period {ref_pair[0]} to {ref_pair[1]} with ±2σ uncertainty bands"
        
        # Case 4: All three are different (original behavior)
        else:
            if has_data_dates:
                description += f" from {data_pair[0]} to {data_pair[1]}"
            if has_ref_dates:
                description += f", reference from {ref_pair[0]} to {ref_pair[1]}"
            if has_std_dates:
                description += f" with ±2σ uncertainty bands computed over {std_pair[0]} to {std_pair[1]}"
        
        description += '.'
        
        self.logger.warning('Description: %s', description)
        return description

    def run(self,
            outputdir='./',
            rebuild=True, 
            dpi=300,
            style=None,
            format='png',
            show=False):
        """
        Unified run method that handles all plotting scenarios.
        
        Args:
            outputdir (str): Output directory to save the plot.
            rebuild (bool): If True, rebuild the plot even if it already exists.
            dpi (int): Dots per inch for the plot.
            style (str): Plotting style. Default is the AQUA style.
            format (str): Format of the plot ('png' or 'pdf'). Default is 'png'.
            show (bool): If True, display the plot interactively.
        """
        self.logger.info('Running PlotLatLonProfiles')

        if self.data_type == 'seasonal':
            return self._run_seasonal(outputdir=outputdir, rebuild=rebuild, dpi=dpi, format=format,
                                      style=style, show=show)
        elif self.data_type == 'longterm':
            return self._run_annual(outputdir=outputdir, rebuild=rebuild, dpi=dpi, format=format,
                                    style=style, show=show)

    def _run_annual(self, outputdir, rebuild, dpi, format, style, show=False):
        """Private method for annual single variable plotting."""
        data_label = self.set_data_labels()
        ref_label = self.set_ref_label()
        description = self.set_description()
        title = self.set_title()
        
        if self.ref_std_data is not None:
            description += " with standard deviation bands"

        fig, _ = self.plot(data_labels=data_label, ref_label=ref_label, title=title,
                           style=style)

        self.save_plot(fig, description=description, rebuild=rebuild,
                       outputdir=outputdir, dpi=dpi, format=format, diagnostic=self.diagnostic_name)
        
        if show:
            plt.show()
            
        plt.close(fig)

        self.logger.info('PlotLatLonProfiles completed successfully')

    def _run_seasonal(self, outputdir, rebuild, dpi, format, style, show=False):
        """Private method for seasonal single variable plotting."""
        data_labels = self.set_data_labels()
        description = self.set_description()
        title = self.set_title()

        fig, _ = self.plot_seasonal_lines(data_labels=data_labels, 
                                          title=title, style=style)

        self.save_plot(fig, description=description, 
                       rebuild=rebuild, outputdir=outputdir, dpi=dpi, format=format, 
                       diagnostic=self.diagnostic_name)
        
        if show:
            plt.show()

        plt.close(fig)
        
        self.logger.info('PlotLatLonProfiles completed successfully')

    def plot_seasonal_lines(self, 
                            data_labels=None, 
                            title=None, 
                            style=None):
        """
        Plot seasonal means using plot_seasonal_lat_lon_profiles.
        Creates a 4-panel plot with DJF, MAM, JJA, SON only.

        Args:
            data_labels (list): List of data labels.
            title (str): Title of the plot.
            style (str): Plotting style. Default is the AQUA style

        Returns:
            fig (matplotlib.figure.Figure): Figure object.
            axs (list): List of axes objects.
        """
        if not self.data or len(self.data) < 4:
            raise ValueError("Seasonal data must contain at least 4 elements: [DJF, MAM, JJA, SON]")
        
        # Use first 4 seasons only (DJF, MAM, JJA, SON)
        seasonal_data_only = self.data[:4]
        seasonal_ref_only = self.ref_data[:4] if self.ref_data and len(self.ref_data) >= 4 else None

        self.logger.debug(f'Plotting {len(seasonal_data_only)} seasons')
        
        return plot_seasonal_lat_lon_profiles(
            seasonal_data=seasonal_data_only,
            ref_data=seasonal_ref_only,
            ref_std_data=self.ref_std_data,
            data_labels=data_labels,
            title=title,
            style=style,
            loglevel=self.loglevel
        )
