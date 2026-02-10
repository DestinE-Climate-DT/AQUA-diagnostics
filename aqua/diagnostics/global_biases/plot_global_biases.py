import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from aqua.core.logger import log_configure
from aqua.core.graphics import plot_single_map, plot_single_map_diff, plot_maps, plot_vertical_profile_diff
from aqua.core.util import get_projection, get_realizations
from aqua.diagnostics.base import OutputSaver
from .stat_global_biases import StatGlobalBiases
from .util import handle_pressure_level

class PlotGlobalBiases: 
    def __init__(self, 
                 diagnostic='globalbiases',
                 save_pdf=True, save_png=True, 
                 dpi=300, outputdir='./',
                 cmap='RdBu_r',
                 loglevel='WARNING'):
        """
        Initialize the PlotGlobalBiases class.

        Args:
            diagnostic (str): Name of the diagnostic.
            save_pdf (bool): Whether to save the figure as PDF.
            save_png (bool): Whether to save the figure as PNG.
            dpi (int): Resolution of saved figures.
            outputdir (str): Output directory for saved plots.
            cmap (str): Colormap to use for the plots.
            loglevel (str): Logging level.
        """
        self.diagnostic = diagnostic
        self.save_pdf = save_pdf
        self.save_png = save_png
        self.dpi = dpi
        self.outputdir = outputdir
        self.cmap = cmap
        self.loglevel = loglevel

        self.logger = log_configure(log_level=loglevel, log_name='Global Biases')

    def _save_figure(self, fig, diagnostic_product, 
                     data, description, var, data_ref=None, 
                     plev=None, **kwargs):
        """
        Handles the saving of a figure using OutputSaver.

        Args:
            fig (matplotlib.Figure): The figure to save.
            data (xarray.Dataset): Dataset.
            data_ref (xarray.Dataset, optional): Reference dataset.
            diagnostic_product (str): Name of the diagnostic product.
            description (str): Description of the figure.
            var (str): Variable name.
            plev (float, optional): Pressure level.
        Keyword Args:
            **kwargs: Additional keyword arguments to be passed to the OutputSaver.
        """
        outputsaver = OutputSaver(
            diagnostic=self.diagnostic,
            catalog=data.AQUA_catalog,
            model=data.AQUA_model,
            exp=data.AQUA_exp,
            model_ref=data_ref.AQUA_model if data_ref else None,
            exp_ref=data_ref.AQUA_exp if data_ref else None,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
            **kwargs
        )

        metadata = {"Description": description}
        extra_keys = {}

        if var is not None:
            extra_keys.update({'var': var})
        if plev is not None:
            extra_keys.update({'plev': plev})

        outputsaver.save_figure(fig, diagnostic_product,
                                extra_keys=extra_keys, metadata=metadata,
                                save_pdf=self.save_pdf, save_png=self.save_png,
                                dpi=self.dpi)

    def _compute_bias_significance(self, data_ts, data_ref_ts, var, plev, alpha):
        """
        Compute the statistical significance of the bias between two datasets using a t-test.
        Args:
            data_ts (xarray.Dataset): Time series dataset for the primary data.
            data_ref_ts (xarray.Dataset): Time series dataset for the reference data.
            var (str): Variable name to analyze.
            plev (float): Pressure level to analyze (if applicable).
            alpha (float): Significance level for the t-test (e.g., 0.05 for 95% confidence).
        Returns:
            xarray.DataArray: A boolean mask indicating where the bias is statistically significant.
        """
        data_ts = handle_pressure_level(data_ts, var, plev, loglevel=self.loglevel)
        data_ref_ts = handle_pressure_level(data_ref_ts, var, plev, loglevel=self.loglevel)

        stat_test = StatGlobalBiases(loglevel=self.loglevel)

        return stat_test.compute_significance_ttest(
            data_ts, data_ref_ts, var, alpha=alpha
        )

    def _add_significance_stippling(self, ax, significance_mask, lat, lon,
                                    stipple_density=3, stipple_size=0.5,
                                    stipple_color='black', invert_mask=False
                                    ):
        """
        Add stippling to indicate statistical significance on a map.

        The function subsamples the significance mask to avoid overcrowding
        and plots small dots (stipples) at grid points where the mask is True.
        Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        significance_mask (xarray.DataArray): Boolean mask indicating significant points.
        lat (xarray.DataArray): Latitude coordinates.
        lon (xarray.DataArray): Longitude coordinates.
        stipple_density (int, optional): Subsampling factor for the mask (e.g., 3 means every 3rd point). Default is 3.
        stipple_size (float, optional): Size of the stipple dots. Default is 0.5.
        stipple_color (str, optional): Color of the stipple dots. Default is 'black'.
        invert_mask (bool, optional): If True, stipple where the mask is False (i.e., non-significant points). Default is False (stippling where significant).
        """

        # Subsample the significance mask along latitude and longitude
        # (e.g. every Nth grid point) to control stippling density
        mask_sub = significance_mask.isel(
            lat=slice(None, None, stipple_density),
            lon=slice(None, None, stipple_density)
        )

        # Extract the corresponding subsampled latitude and longitude coordinates
        lat_sub = mask_sub.lat
        lon_sub = mask_sub.lon

        # Create 2D coordinate grids for plotting
        # This maps each grid point to its geographic location
        lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)

        # Optionally invert the mask:
        # - False (default): stipple where differences ARE significant
        # - True: stipple where differences are NOT significant
        mask_to_plot = ~mask_sub if invert_mask else mask_sub

        # Plot stippling using a scatter plot:
        # dots are placed only at grid points where mask_to_plot is True
        ax.scatter(
            lon_mesh[mask_to_plot.values],
            lat_mesh[mask_to_plot.values],
            s=stipple_size,
            c=stipple_color,
            transform=ccrs.PlateCarree(),
            alpha=0.6,
            linewidths=0
        )


    def plot_climatology(self, data, var, plev=None, proj='robinson', proj_params={}, vmin=None, vmax=None, cbar_label=None):
        """
        Plots the climatology map for a given variable and time range.

        Args:
            data (xarray.Dataset): Climatology dataset to plot.
            var (str): Variable name.
            plev (float, optional): Pressure level to plot (if applicable).
            proj (string, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection (e.g., {'central_longitude': 0}).
            vmin (float, optional): Minimum color scale value.
            vmax (float, optional): Maximum color scale value.
            cbar_label (str, optional): Label for the colorbar.

        Returns:
            tuple: Matplotlib figure and axis objects.
        """
        self.logger.info('Plotting climatology.')
        
        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        if data is None:
            return None

        realization = get_realizations(data)
        proj = get_projection(proj, **proj_params)
        
        title = (f"Climatology of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}" 
                + (f" at {int(plev / 100)} hPa" if plev else ""))

        fig, ax = plot_single_map(
            data[var],
            return_fig=True,
            title=title,
            title_size=18,
            vmin=vmin,
            vmax=vmax,
            proj=proj,
            loglevel=self.loglevel,
            cbar_label=cbar_label,
            cmap=self.cmap
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        description = (
            f"Spatial map of the climatology {data[var].attrs.get('long_name', var)}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''}"
            f" from {data.startdate} to {data.enddate} "
            f"for the {data.AQUA_model} model, experiment {data.AQUA_exp}."
        )

        self._save_figure(fig=fig, diagnostic_product='annual_climatology',
                          data=data, description=description, var=var, plev=plev, realization=realization)

    def plot_bias(self, data, data_ref, var, plev=None, proj='robinson', proj_params={}, vmin=None, vmax=None, cbar_label=None, area=None, show_stats=False,
                  data_timeseries=None, data_ref_timeseries=None, 
                  show_significance=False, significance_alpha=0.05,
                  stipple_density=3, stipple_size=0.5, invert_stippling=False):
        """
        Plots the bias map between two datasets.

        Args:
            data (xarray.Dataset): Primary dataset.
            data_ref (xarray.Dataset): Reference dataset.
            var (str): Variable name.
            plev (float, optional): Pressure level.
            proj (str, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            cbar_label (str, optional): Label for the colorbar.
            area (xr.DataArray, optional): Grid cell areas for computing weighted statistics.
            show_stats (bool, optional): Whether to show statistical information on the plot.
        """
        self.logger.info('Plotting global biases.')

        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        data_ref = handle_pressure_level(data_ref, var, plev, loglevel=self.loglevel)

        realization = get_realizations(data)

        sym = vmin is None or vmax is None

        proj = get_projection(proj, **proj_params)

        title = (f"Global bias of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}\n"
                 f"relative to {data_ref.AQUA_model} climatology"
                 + (f" at {int(plev / 100)} hPa" if plev else ""))

        fig, ax = plot_single_map_diff(
            data=data[var], 
            data_ref=data_ref[var],
            return_fig=True,
            contour=True, 
            title=title,
            title_size=18,
            sym=sym,
            proj=proj,
            vmin_fill=vmin, 
            vmax_fill=vmax,
            cbar_label=cbar_label,
            cmap=self.cmap,
            loglevel=self.loglevel
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")


        # Add significance stippling if requested
        if show_significance and data_timeseries is not None:

            self.logger.info('Computing statistical significance for bias.')

            significance_mask = self._compute_bias_significance(
            data_timeseries, data_ref_timeseries, var, plev=plev, alpha=significance_alpha)

            # Add stippling
            lat = data[var].coords.get('lat', data[var].coords.get('latitude'))
            lon = data[var].coords.get('lon', data[var].coords.get('longitude'))
            
            self._add_significance_stippling(
                ax, significance_mask, lat, lon,
                stipple_density=stipple_density,
                stipple_size=stipple_size,
                invert_mask=invert_stippling
            )
           
            pct_sig = significance_mask.attrs.get('percent_significant', 0)
            n_samples = significance_mask.attrs.get('n_samples_model', 'unknown')
            self.logger.info(f'Added significance stippling: {pct_sig:.1f}% of points are significant.')

            ax.text(
                0.99, 0.01,
                f"Stippling: p < {significance_alpha}\n"
                f"Welch t-test, N = {n_samples}",
                f"Significance: {pct_sig:.1f}% of points",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

        #  Add statistics to the plot if requested
        if show_stats:
            self.logger.debug('Computing bias statistics.')
            if area is None:
                self.logger.warning('Grid areas not provided, unweighted statistics will be computed.')
            bs = StatGlobalBiases(loglevel=self.loglevel)
            stats = bs.compute_bias_statistics(
                data=data,
                data_ref=data_ref,
                var=var,
                area=area
            )
            mean_bias = float(stats.mean_bias.values)
            rmse = float(stats.rmse.values)
            units = data[var].attrs.get('units', '')
            stats_text = f"Mean: {mean_bias:{'.2g'}}" + (f" {units}" if units else "")
            stats_text += f"\nRMSE: {rmse:{'.2g'}}" + (f" {units}" if units else "")

            x, y, ha, va = (0.02, 0.87, 'left', 'center')
            # Add text to figure
            fig.text(x, y, stats_text,
                    fontsize=12,
                    ha=ha, va=va,
                    transform=fig.transFigure,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
            
            self.logger.info(f"Added statistics to plot: Mean={mean_bias:.2g}, RMSE={rmse:.2g}")
        description = (
            f"Spatial map of global bias of {data[var].attrs.get('long_name', var)}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''}"
            f" from {data.startdate} to {data.enddate}"
            f" for the {data.AQUA_model} model, experiment {data.AQUA_exp}, with {data_ref.AQUA_model}"
            f" from {data_ref.startdate} to {data_ref.enddate} used as reference data."
        )

        self._save_figure(fig=fig, diagnostic_product='bias', data=data, data_ref=data_ref,
                          description=description, var=var, plev=plev, realization=realization)


    def plot_seasonal_bias(self, data, data_ref, var, plev=None, proj='robinson', proj_params={}, vmin=None, vmax=None, cbar_label=None):
        """
        Plots seasonal biases for each season (DJF, MAM, JJA, SON).

        Args:
            data (xarray.Dataset): Primary dataset.
            data_ref (xarray.Dataset): Reference dataset.
            var (str): Variable name.
            plev (float, optional): Pressure level.
            proj (str, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            cbar_label (str, optional): Label for the colorbar.

        Returns:
            matplotlib.figure.Figure: The resulting figure.
        """
        self.logger.info('Plotting seasonal biases.')

        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        data_ref = handle_pressure_level(data_ref, var, plev, loglevel=self.loglevel)

        realization = get_realizations(data)

        season_list = ['DJF', 'MAM', 'JJA', 'SON']
        sym = vmin is None or vmax is None

        title = (f"Seasonal bias of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}\n"
                 f"relative to {data_ref.AQUA_model} climatology"
                 + (f" at {int(plev / 100)} hPa" if plev else ""))

        plot_kwargs = {
            'maps': [data[var].sel(season=season) - data_ref[var].sel(season=season) for season in season_list],
            'proj': get_projection(proj, **proj_params),
            'return_fig': True,
            'title': title,
            'title_size': 18,
            'titles': season_list,
            'titles_size': 16,
            'figsize':(10, 8),
            'contour': True,
            'sym': sym,
            'cbar_label': cbar_label,
            'cmap': self.cmap,
            'loglevel': self.loglevel
        }

        if vmin is not None:
            plot_kwargs['vmin'] = vmin
        if vmax is not None:
            plot_kwargs['vmax'] = vmax

        fig = plot_maps(**plot_kwargs)

        description = (
            f"Seasonal bias map of {data[var].attrs.get('long_name', var)}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''}"
            f" for the {data.AQUA_model} model, experiment {data.AQUA_exp},"
            f" using {data_ref.AQUA_model} as reference data."
            f" The bias is computed for each season over the period from {data.startdate} to {data.enddate} for the model" 
            f" and from {data_ref.startdate} to {data_ref.enddate} for the reference data."
        )

        self._save_figure(fig=fig, diagnostic_product='seasonal_bias', data=data, data_ref=data_ref,
                          description=description, var=var, plev=plev, realization=realization)


    def plot_vertical_bias(self, data, data_ref, var, plev_min=None, plev_max=None, 
                           vmin=None, vmax=None, vmin_contour=None, vmax_contour=None, nlevels=18):
        """
        Calculates and plots the vertical bias between two datasets.

        Args:
            data (xarray.Dataset): Dataset to analyze.
            data_ref (xarray.Dataset): Reference dataset for comparison.
            var (str): Variable name to analyze.
            plev_min (float, optional): Minimum pressure level.
            plev_max (float, optional): Maximum pressure level.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            vmin_contour (float, optional): Minimum contour value.
            vmax_contour (float, optional): Maximum contour value.
            nlevels (int, optional): Number of contour levels for the plot.
        """
        self.logger.info('Plotting vertical biases for variable: %s', var)

        realization = get_realizations(data)

        title = (
            f"Vertical bias of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}\n"
            f"relative to {data_ref.AQUA_model} climatology\n"
        )

        description = (
            f"Vertical bias plot of {data[var].attrs.get('long_name', var)} across pressure levels from {data.startdate} to {data.enddate}"
            f" for the {data.AQUA_model} model, experiment {data.AQUA_exp}, with {data_ref.AQUA_model} from {data_ref.startdate} to {data_ref.enddate}"
            f" used as reference data."
        )

        fig, ax = plot_vertical_profile_diff(
            data=data[var].mean(dim='lon'),
            data_ref=data_ref[var].mean(dim='lon'),
            var=var,
            lev_min=plev_min,
            lev_max=plev_max,
            vmin=vmin,
            vmax=vmax,
            vmin_contour=vmin_contour,
            vmax_contour=vmax_contour,
            logscale=True,
            add_contour=True, 
            cmap=self.cmap,
            nlevels=nlevels,
            title=title,
            title_size=18,
            return_fig=True,
            loglevel=self.loglevel
        )

        self._save_figure(fig=fig, diagnostic_product='vertical_bias', data=data, data_ref=data_ref,
                          description=description, var=var, realization=realization)

        self.logger.info("Vertical bias plot completed successfully.")
