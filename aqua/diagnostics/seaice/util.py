"""Utility for the sea ice plotting module"""

import pandas as pd


def filter_region_list(regions_dict, regions_list, domain, logger, valid_domains=None):
    """Filters a list of string regions based on config_file defined coords values and specified domain.
    This function checks if regions fall within the appropriate hemisphere based on their latitude bounds.

    Args:
        regions_dict (dict): Dictionary containing region definitions.
        regions_list (list): List of region names to be filtered.
        domain (str): Domain to filter regions by. Must be one of the valid domains, e.g., 'nh' or 'sh'.
        logger (logging.Logger): Logger instance for logging messages.
        valid_domains (list, optional): List of valid domain strings. Defaults to ['nh', 'sh'].

    Returns:
        list or str: Filtered list of region names. If exactly one region is valid, returns a single string.
    """
    if not valid_domains:
        valid_domains = ["nh", "sh"]

    if domain not in valid_domains:
        raise ValueError(f"Invalid domain '{domain}'. Valid domains are: {valid_domains}")

    filtered_regions = []
    for r in regions_list:
        if r in regions_dict["regions"].keys():
            region_info = regions_dict["regions"][r]
            lat_limits = region_info.get("lat_limits", [])

            if len(lat_limits) >= 2:
                min_lat, max_lat = lat_limits[0], lat_limits[1]

            if domain == "nh" and min_lat >= 0:  # Northern hemisphere
                filtered_regions.append(r)
            elif domain == "sh" and max_lat <= 0:  # Southern hemisphere
                filtered_regions.append(r)
            else:
                logger.debug(
                    f"Region '{r}' doesn't meet the data domain criteria for {domain}, not including in regions_list."
                )
        else:
            logger.error(f"No region '{r}' defined in regions_dict from yaml. Check this mismatch.")

    return filtered_regions


def _fmt_time_attr(data, attr_name):
    """Format a single date attribute as YYYY-MM for captions."""
    date = data.attrs.get(attr_name)
    return pd.Timestamp(date).strftime("%Y-%m") if date is not None else f"No {attr_name} found"


def extract_dates(data):
    """
    Extracts start and end dates from data attributes, formatted as 'YYYY-MM'.

    Args:
        data (xr.DataArray | xr.Dataset): Input object holding the `AQUA_startdate` / `AQUA_enddate` attributes.

    Returns:
        tuple[str, str]: `(start_date, end_date)` formatted as `YYYY-MM`.
    """
    return _fmt_time_attr(data, "AQUA_startdate"), _fmt_time_attr(data, "AQUA_enddate")


def extract_std_period_dates(data):
    """
    Extract the time span from ``AQUA_std_startdate`` and ``AQUA_std_enddate``.

    Args:
        data (xr.DataArray | xr.Dataset): Standard deviation computed by ``SeaIce``.

    Returns:
        tuple[str, str]: `(start_date, end_date)` formatted as `YYYY-MM`.
    """
    return _fmt_time_attr(data, "AQUA_std_startdate"), _fmt_time_attr(data, "AQUA_std_enddate")


def _check_list_regions_type(regions_to_plot, logger=None):
    """Validate regions_to_plot is a list of strings.

    Args:
        regions_to_plot (list[str] | None): Regions requested for plotting, or `None` to plot all regions.
        logger (logging.Logger | None): Logger used for warning messages.

    Returns:
        list[str] | None: The validated list of region names, or `None`.

    Raises:
        TypeError: If `regions_to_plot` is not a list of strings.
    """
    if regions_to_plot is None:
        logger.warning("Expected regions_to_plot to be a list, but got None. Plotting all available regions in data.")
        return None

    if not isinstance(regions_to_plot, list):
        raise TypeError(f"Expected regions_to_plot to be a list, but got {type(regions_to_plot).__name__}.")

    if not all(isinstance(region, str) for region in regions_to_plot):
        invalid_types = [type(region).__name__ for region in regions_to_plot]
        raise TypeError(f"Expected a list of strings, but found element types: {invalid_types}.")
    return regions_to_plot
