import xarray as xr
import numpy as np
from scipy import stats
from aqua.core.logger import log_configure
from aqua.core.fldstat import FldStat
from .util import handle_pressure_level


xr.set_options(keep_attrs=True)


class StatGlobalBiases:
    """
    Class for computing bias statistics between model and reference data.
    It works directly with xarray datasets and includes statistical significance testing.

    Args:
        loglevel (str): Log level. Default is 'WARNING'.
    """
    def __init__(self, loglevel: str = 'WARNING'):

        self.logger = log_configure(log_level=loglevel, log_name='Bias Statistics')
        self.loglevel = loglevel

    def compute_bias_statistics(self,
                                data: xr.Dataset,
                                data_ref: xr.Dataset,
                                var: str,
                                area: xr.DataArray = None,
                                ) -> xr.Dataset:
        """
        Compute global mean bias and RMSE between model and reference data.

        Args:
            data (xr.Dataset): Model climatology dataset.
            data_ref (xr.Dataset): Reference climatology dataset.
            var (str): Variable name.
            area (xr.DataArray, optional): Grid cell areas for weighted statistics.
                                            If None, unweighted statistics will be computed.
        Returns: 
            xr.Dataset: Dataset containing mean bias and RMSE.

        """
        self.logger.info(f'Computing bias statistics for variable {var}.')

        if data is None or data_ref is None:
            raise ValueError("Data or reference data is None after pressure level handling.")

        # Compute bias
        bias = data[var] - data_ref[var]
        
        fldstat = FldStat(area=area, loglevel=self.loglevel)

        # Compute mean bias
        self.logger.debug('Computing area-weighted mean bias.')
        mean_bias = fldstat.fldstat(
            bias,
            stat='mean',
        )

        # Compute RMSE: sqrt(mean(bias^2))
        self.logger.debug('Computing RMSE.')
        bias_squared = bias ** 2
        mean_squared_error = fldstat.fldstat(
            bias_squared,
            stat='mean'
        )
        rmse = np.sqrt(mean_squared_error)

        stats = xr.Dataset({
            'mean_bias': mean_bias,
            'rmse': rmse
        })

        self.logger.info(f'Mean bias: {float(mean_bias.values):.4e} {data[var].attrs.get("units", "")}')
        self.logger.info(f'RMSE: {float(rmse.values):.4e} {data[var].attrs.get("units", "")}')

        return stats
    
    def compute_temporal_means(self,
                               data: xr.Dataset,
                               var: str,
                               freq: str = 'year',
                               season: str = None) -> xr.DataArray:
        """
        Compute temporal means (yearly or seasonal).

        Args:
            data (xr.Dataset): Input dataset with time dimension.
            var (str): Variable name.
            freq (str): Frequency for grouping: 'year' or 'season'. Default is 'year'.
            season (str, optional): If provided, select only this season (DJF, MAM, JJA, SON).
                                   Only used when freq='season'.

        Returns:
            xr.DataArray: Array with temporal means. Shape will be (time, lat, lon) where
                         time represents years or seasons.
        """
        self.logger.info(f'Computing temporal means with frequency: {freq}')

        if 'time' not in data[var].dims:
            raise ValueError(f"Variable {var} does not have a 'time' dimension.")

        if freq == 'year':
            yearly_means = data[var].groupby('time.year').mean('time')
            yearly_means = yearly_means.rename({'year': 'time'})
            self.logger.info(f'Computed {len(yearly_means.time)} yearly means.')
            return yearly_means

        elif freq == 'season':
            seasonal_means = data[var].groupby('time.season').mean('time')
            
            if season is not None:
                if season not in ['DJF', 'MAM', 'JJA', 'SON']:
                    raise ValueError(f"Season must be one of: DJF, MAM, JJA, SON. Got: {season}")
                seasonal_means = seasonal_means.sel(season=season)
                self.logger.info(f'Selected season: {season}')
            
            return seasonal_means

        else:
            raise ValueError(f"freq must be 'year' or 'season'. Got: {freq}")


    def ttest_at_grid_point(self, model_vals, ref_vals, min_samples: int = 3):
        """Perform t-test at a single grid point.
        Args:
            model_vals (np.ndarray): 1D array of model values at a grid point.
            ref_vals (np.ndarray): 1D array of reference values at the same grid point.
            min_samples (int): Minimum number of samples required to perform the t-test. Default is 3.
        Returns:
            float: p-value from the t-test.
        """
        # Remove NaN values
        model_clean = model_vals[np.isfinite(model_vals)]
        ref_clean = ref_vals[np.isfinite(ref_vals)]
        
        if len(model_clean) < min_samples or len(ref_clean) < min_samples:
            return np.nan
        
        # Perform t-test
        _, p_value = stats.ttest_ind(model_clean, ref_clean, equal_var=False)
        return p_value


    def compute_significance_ttest(self,
                                   data: xr.Dataset,
                                   data_ref: xr.Dataset,
                                   var: str,
                                   alpha: float = 0.05,
                                   freq: str = 'year',
                                   season: str = None,
                                   min_samples: int = 3) -> xr.DataArray:
        """
        Compute statistical significance of bias using two-sample t-test.
        
        Performs a two-sided t-test at each grid point to determine if the difference
        between model and reference data is statistically significant.

        Args:
            data (xr.Dataset): Model dataset with time dimension.
            data_ref (xr.Dataset): Reference dataset with time dimension.
            var (str): Variable name.
            alpha (float): Significance level (default: 0.05 for 95% confidence).
            freq (str): Frequency for temporal grouping: 'year' or 'season'. Default is 'year'.
            season (str, optional): Season to analyze (only for freq='season').
            min_samples (int): Minimum number of samples required to perform test. Default is 3.

        Returns:
            xr.DataArray: Boolean array where True indicates statistically significant differences.
                         Same spatial dimensions as input data.
        """
        self.logger.info(f'Computing statistical significance using t-test (alpha={alpha}).')

        # Get temporal means
        data_temporal = self.compute_temporal_means(data, var, freq=freq, season=season)
        data_ref_temporal = self.compute_temporal_means(data_ref, var, freq=freq, season=season)

        # Check if we have enough samples
        n_samples = len(data_temporal.time) if 'time' in data_temporal.dims else len(data_temporal.season)
        n_samples_ref = len(data_ref_temporal.time) if 'time' in data_ref_temporal.dims else len(data_ref_temporal.season)
        
        self.logger.info(f'Number of samples - Model: {n_samples}, Reference: {n_samples_ref}')

        if n_samples < min_samples or n_samples_ref < min_samples:
            self.logger.warning(
                f'Insufficient samples for t-test. Model: {n_samples}, Reference: {n_samples_ref}. '
                f'Minimum required: {min_samples}. Returning all False.'
            )
            # Return array of False (not significant) with same shape as spatial dimensions
            return xr.DataArray(
                np.zeros(data[var].isel(time=0).shape, dtype=bool),
                coords={k: v for k, v in data[var].isel(time=0).coords.items() if k != 'time'},
                dims=[d for d in data[var].dims if d != 'time']
            )

        # Rechunk along time dimension for dask compatibility
        # This is needed because apply_ufunc with dask='parallelized' requires
        # core dimensions to be in a single chunk
                
        # Get time dimension name
        time_dim = 'time' if 'time' in data_temporal.dims else 'season'
        
        if time_dim in data_temporal.dims:
            self.logger.debug(f'Rechunking data along {time_dim} dimension.')
            data_temporal = data_temporal.chunk({time_dim: -1})
            data_ref_temporal = data_ref_temporal.chunk({time_dim: -1})

        # Perform t-test at each grid point
        # Use scipy.stats.ttest_ind for independent samples
        self.logger.debug('Performing t-test at each grid point.')

        # Apply t-test across all grid points
        p_values = xr.apply_ufunc(
            self.ttest_at_grid_point,
            data_temporal,
            data_ref_temporal,
            input_core_dims=[[time_dim], [time_dim]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            kwargs={"min_samples": min_samples},
        )
        # Determine significance (True where p-value < alpha)
        is_significant = p_values < alpha
        
        # Count significant points
        n_significant = int(is_significant.sum().values)
        n_total = int(np.prod(is_significant.shape))
        pct_significant = 100 * n_significant / n_total if n_total > 0 else 0
        
        self.logger.info(
            f'Statistical significance test completed: '
            f'{n_significant}/{n_total} points ({pct_significant:.1f}%) are significant at alpha={alpha}.'
        )

        # Add metadata
        is_significant.attrs.update({
            'long_name': 'Statistical significance of bias',
            'description': f'Two-sample t-test with alpha={alpha}',
            'alpha': alpha,
            'n_samples_model': n_samples,
            'n_samples_reference': n_samples_ref,
            'n_significant_points': n_significant,
            'percent_significant': pct_significant
        })

        return is_significant
