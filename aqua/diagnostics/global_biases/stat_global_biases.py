import xarray as xr
import numpy as np
from aqua.core.logger import log_configure
from aqua.core.fldstat import FldStat
from .util import handle_pressure_level


xr.set_options(keep_attrs=True)


class StatGlobalBiases:
    """
    Class for computing bias statistics between model and reference data.
    It works directly with xarray datasets.

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