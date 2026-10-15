"""Statistics needed by the Taylor diagram."""

import numpy as np
import xarray as xr

from aqua.core.logger import log_configure

xr.set_options(keep_attrs=True)


def compute_taylor_statistics(
    data: xr.DataArray,
    data_ref: xr.DataArray,
    weights: xr.DataArray | None = None,
    dims: str | list[str] | None = None,
    loglevel: str = "WARNING",
) -> xr.Dataset:
    """
    Compute the statistics summarised by a Taylor diagram between a field and its reference.

    The statistics are weighted and computed over ``dims``, any other dimension is kept so that, for example,
    one set of statistics per season can be computed at once. Points where either field is missing are
    excluded from both, so that the two fields are always compared over the same domain. The standard
    deviations are population ones, which is what makes the Taylor diagram geometry exact:
    ``crmse**2 = std**2 + std_ref**2 - 2 * std * std_ref * corr``.

    Args:
        data (xr.DataArray): The field to evaluate.
        data_ref (xr.DataArray): The reference field, on the same grid as data.
        weights (xr.DataArray, optional): The weights, typically the cell area. Uniform if not provided.
        dims (str or list of str, optional): The dimensions over which the statistics are computed.
            Default is all the dimensions of data.
        loglevel (str): Logging level. Default is 'WARNING'.

    Returns:
        xr.Dataset: With the variables std, std_ref, corr, crmse, bias, rmse and the normalised
            std_norm and crmse_norm, divided by the reference standard deviation.

    Raises:
        ValueError: If data and data_ref are not on the same grid along dims.
    """
    logger = log_configure(loglevel, "Taylor statistics")

    dims = list(data.dims) if dims is None else [dims] if isinstance(dims, str) else list(dims)
    data_ref = _match_grid(data=data, data_ref=data_ref, dims=dims)

    if weights is None:
        weights = xr.ones_like(data)
    weights = weights.fillna(0)

    # Joint mask: a point missing in one of the two fields is excluded from both
    valid = data.notnull() & data_ref.notnull()
    weights = xr.where(valid, weights, 0)
    weights_sum = weights.sum(dim=dims)

    if (weights_sum == 0).any():
        logger.warning("No valid point in common between the field and the reference for some of the statistics")

    def wmean(field):
        return (field.fillna(0) * weights).sum(dim=dims) / weights_sum

    mean = wmean(data)
    mean_ref = wmean(data_ref)
    anom = data - mean
    anom_ref = data_ref - mean_ref

    std = np.sqrt(wmean(anom**2))
    std_ref = np.sqrt(wmean(anom_ref**2))
    corr = wmean(anom * anom_ref) / (std * std_ref)
    crmse = np.sqrt(wmean((anom - anom_ref) ** 2))
    rmse = np.sqrt(wmean((data - data_ref) ** 2))

    units = data.attrs.get("units")
    stats = {
        "std": (std, "Standard deviation", units),
        "std_ref": (std_ref, "Standard deviation of the reference", units),
        "corr": (corr, "Correlation with the reference", "1"),
        "crmse": (crmse, "Centred root mean square difference", units),
        "bias": (mean - mean_ref, "Mean bias", units),
        "rmse": (rmse, "Root mean square difference", units),
        "std_norm": (std / std_ref, "Normalised standard deviation", "1"),
        "crmse_norm": (crmse / std_ref, "Normalised centred root mean square difference", "1"),
    }

    statistics = xr.Dataset()
    for name, (values, long_name, unit) in stats.items():
        values = values.copy()
        values.attrs = {"long_name": long_name}
        if unit is not None:
            values.attrs["units"] = unit
        statistics[name] = values

    return statistics


def _match_grid(data: xr.DataArray, data_ref: xr.DataArray, dims: list[str]) -> xr.DataArray:
    """
    Check that data_ref is on the same grid as data along dims and give it the coordinates of data.

    Coordinates are compared with a tolerance, so that grids which differ only by floating point noise,
    e.g. because of different precisions, are considered the same.

    Args:
        data (xr.DataArray): The field to evaluate.
        data_ref (xr.DataArray): The reference field.
        dims (list of str): The dimensions to check.

    Returns:
        xr.DataArray: data_ref with the coordinates of data along dims.

    Raises:
        ValueError: If a dimension is missing, has a different size or different coordinate values.
    """
    hint = "Data and reference must be on the same grid, regrid them to a common one (e.g. regrid: r100)"
    for dim in dims:
        if dim not in data.dims or dim not in data_ref.dims:
            raise ValueError(f"Dimension '{dim}' missing in data or reference. {hint}")
        if data.sizes[dim] != data_ref.sizes[dim]:
            raise ValueError(
                f"Dimension '{dim}' has size {data.sizes[dim]} in data, {data_ref.sizes[dim]} in reference. {hint}"
            )
        if dim in data.coords and dim in data_ref.coords:
            if not np.allclose(data[dim].values, data_ref[dim].values):
                raise ValueError(f"Coordinate '{dim}' differs between data and reference. {hint}")
            data_ref = data_ref.assign_coords({dim: data[dim].values})
    return data_ref
