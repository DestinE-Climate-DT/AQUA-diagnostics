"""Utility functions for the LatLonProfiles CLI."""


def load_var_config(config_dict, var, diagnostic="lat_lon_profiles"):
    """Load variable configuration from config dictionary.

    Args:
        config_dict (dict): Configuration dictionary.
        var (str or dict): Variable name or inline variable configuration dictionary.
        diagnostic (str): Diagnostic name.

    Returns:
        tuple: (var_config dict, regions list)
    """
    params = config_dict.get("diagnostics", {}).get(diagnostic, {}).get("params", {})
    default_params = params.get("default", {})

    if isinstance(var, dict):
        var_name = var.get("name")
        var_specific = params.get(var_name, {}) if var_name else {}
        var_config = {**default_params, **var_specific, **var}
    else:
        var_name = var
        var_specific = params.get(var_name, {})
        var_config = {**default_params, **var_specific}
        var_config.setdefault("name", var_name)

    regions = var_config.get("regions", [None])

    return var_config, regions
