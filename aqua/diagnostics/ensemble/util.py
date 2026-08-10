"""
Utility functions for the ensemble class
"""

import gc
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr

from aqua import Reader
from aqua.core.configurer import ConfigPath
from aqua.core.exceptions import NoDataError
from aqua.core.logger import log_configure
from aqua.diagnostics.base import OutputSaver

def reader_retrieve_and_merge(
    filenames: list[str] = None,
    variable: str = None,
    ens_dim: str = "ensemble",
    catalog_list: list[str] = None,
    model_list: list[str] = None,
    exp_list: list[str] = None,
    source_list: list[str] = None,
    reader_kwargs: dict = None,
    realization: list = None,
    region: str = None,
    lon_limits: float = None,
    lat_limits: float = None,
    startdate: str = None,
    enddate: str = None,
    regrid: str = None,
    areas: bool = False,
    fix: bool = True,
    freq: str = None,
    loglevel: str = "WARNING",
):
    """
    Retrieve, merge, and slice datasets from multiple models, experiments, and sources.

    This function uses the AQUA Reader class to load data for a specified variable
    from multiple catalogs, models, experiments, and sources. Individual realizations
    are loaded, optionally subset by spatial (lon/lat) or temporal (start/end date)
    constraints, and concatenated along a specified ensemble dimension. The final
    merged dataset contains all requested ensemble members with appropriate metadata.

    Args:
        filenames (list[str], optional): List of specific filenames to retrieve. Defaults to None.
        variable (str, optional): Name of the variable to retrieve. Defaults to None.
        ens_dim (str, optional): Name of the ensemble dimension for concatenation. Defaults to "ensemble".
        catalog_list (list[str], optional): List of AQUA catalogs to retrieve data from. Defaults to None.
        model_list (list[str], optional): List of models corresponding to catalogs and experiments. Defaults to None.
        exp_list (list[str], optional): List of experiments corresponding to models and sources. Defaults to None.
        source_list (list[str], optional): List of sources corresponding to models and experiments. Defaults to None.
        reader_kwargs (dict, optional): Additional keyword arguments to pass to the AQUA Reader. Defaults to None.
        realization (list[str], optional): List specifying realizations per model. Defaults to None.
        region (str, optional): Region for zonal or spatial selections. Defaults to None.
        lon_limits (float, optional): Longitude limits for spatial subsetting. Defaults to None.
        lat_limits (float, optional): Latitude limits for spatial subsetting. Defaults to None.
        startdate (str, optional): Start date for temporal subsetting. Defaults to None.
        enddate (str, optional): End date for temporal subsetting. Defaults to None.
        regrid (str, optional): Grid to reproject data onto. Defaults to None.
        areas (bool, optional): Whether to calculate area-weighted values. Defaults to False.
        fix (bool, optional): Apply data fixes if necessary. Defaults to True.
        freq (str, optional): Temporal frequency to filter or retrieve. Defaults to None.
        loglevel (str, optional): Logging level for messages. Defaults to "WARNING".

    Returns:
        xarray.Dataset: Merged dataset containing all requested ensemble members,
        concatenated along `ens_dim` with metadata including description, variable,
        and ensemble member labels. Returns None if no input parameters are provided.

    Raises:
        RuntimeError: If no datasets are successfully retrieved from AQUA Reader.

    Notes:
        - If all catalog_list, model_list, exp_list, and source_list are None or empty,
          the function returns None.
        - Handles missing or default realizations by using ["r1"].
        - Automatically frees memory after processing individual datasets.

    TODO:
        - Add support for additional spatial selections beyond lon/lat slices.
        - Improve error handling and reporting for missing datasets.
        - Add option to automatically regrid or interpolate datasets.
        - Include caching mechanism to avoid repeated reads from the same catalog/model/exp/source.
    """

    # TODO:
    # reader_kwargs is assigned as an empty dictionary.
    # Need to test its implementation in the 'reader_retrieve_and_merge' function
    logger = log_configure(log_name="reader_retrieve_and_merge", log_level=loglevel)
    logger.info("Loading and merging the ensemble dataset using the Reader class")

    if all(not v for v in [filenames, catalog_list, model_list, exp_list, source_list]):
        logger.warning("All of catalog, model, exp, and source are None or empty. Exiting reader_retrieve_and_merge.")
        return None
    # Ensure consistent list types
    if isinstance(catalog_list, str):
        catalog_list = [catalog_list]
    if isinstance(model_list, str):
        model_list = [model_list]
    if isinstance(exp_list, str):
        exp_list = [exp_list]
    if isinstance(source_list, str):
        source_list = [source_list]
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(realization, str):
        realization = [realization]

    if realization is None: realization = ["r1"]
    
    model_data_list = []
    # Need this for return
    merged_dataset = None

    # Loop through each (catalog, model, exp, source) combination
    if not filenames:
        for cat_i, model_i, exp_i, source_i in zip(catalog_list, model_list, exp_list, source_list):
            logger.info(f"Processing: catalog={cat_i}, model={model_i}, exp={exp_i}, source={source_i}")
            # loop over realization(s)
            for r in realization:
                data = reader_loop_over_realizations(
                    catalog=cat_i,
                    model=model_i,
                    exp=exp_i,
                    source=source_i,
                    realization_list=r,
                    areas=areas,
                    fix=fix,
                    reader_kwargs=reader_kwargs,
                    freq=freq,
                    startdate=startdate,
                    enddate=enddate,
                    loglevel=loglevel,
                )
                logger.info(f"Loaded {variable} for {model_i}, {exp_i}, realization={r}")

                # Spatial selection
                if lon_limits and lat_limits:
                    if "lon" in data.dims and "lat" in data.dims:
                        data = data.sel(lon=slice(*lon_limits), lat=slice(*lat_limits))
                    else:
                        logger.debug(f"Dataset for {model_i}-{r} has no lon/lat dims, skipping spatial subset.")

                # Temporal selection (only if time dimension exists)
                if "time" in data.dims and (startdate or enddate):
                    data = data.sel(time=slice(startdate, enddate))
                elif "time" not in data.dims and (startdate or enddate):
                    logger.debug(f"Dataset for {model_i}-{r} has no time dimension.")

                if data is None:
                    continue
                # Add ensemble label
                ens_label = f"{model_i}_{exp_i}"
                data = data.expand_dims({ens_dim: [ens_label]})
                
                model_data_list.append(data)
    
        if not model_data_list:
            logger.warning(f"No realizations loaded using Reader. Skipping...")

    elif filenames:
        # For now Single model only using AQUA Reader backend 
        for filename, r in zip(filenames, realization):
            logger.info(f"Processing: file {filename}")
            try:
                # Retrieve the data using AQUA Reader
                reader = Reader(
                    path=filename,
                    startdate=startdate,
                    enddate=enddate,
                    reader_kwargs=reader_kwargs,
                    areas=areas,
                    fix=fix,
                )

                data = reader.retrieve(var=variable)

                logger.info(f"Loaded {variable} for {filename}")
                # Spatial selection
                if lon_limits and lat_limits:
                    if "lon" in data.dims and "lat" in data.dims:
                        data = data.sel(lon=slice(*lon_limits), lat=slice(*lat_limits))
                    else:
                        logger.debug(f"Dataset for filename {filename} has no lon/lat dims, skipping spatial subset.")

                # Temporal selection (only if time dimension exists)
                if "time" in data.dims and (startdate or enddate):
                    data = data.sel(time=slice(startdate, enddate))
                elif "time" not in data.dims and (startdate or enddate):
                    logger.debug(f"Dataset for filename {filename}  has no time dimension.")

                # Add ensemble label
                ens_label = f"{r}"
                data = data.expand_dims({ens_dim: [ens_label]})

                model_data_list.append(data)

            except Exception as e:
                logger.warning(f"Skipping filename {filename} due to error: {e}")
                continue

        if not model_data_list:
            logger.warning(f"No realizations loaded using Reader-Backend. Skipping...")

    # Merge across all models
    if model_data_list:
        merged_dataset = xr.concat(model_data_list, dim=ens_dim, combine_attrs="override")
        logger.debug(f"Merged dataset {merged_dataset}")

        # Free up memory from individual realizations
        for data in model_data_list:
            data.close() if hasattr(data, "close") else None
        del model_data_list
        gc.collect()
        logger.info("Memory successfully freed.")
        
        # Adding metadata
        merged_dataset.attrs.update(
            {
                "description": "Merged data for AQUA ensemble diagnostics across models, experiments, and realizations.",
                "variable": variable,
                "ensemble_members": list(merged_dataset[ens_dim].values),
            }
        )

    return merged_dataset

def reader_loop_over_realizations(
    variable: str = None,
    ens_dim: str = "ensemble",
    catalog: str = None,
    model: str = None,
    exp: str = None,
    source: str = None,
    areas: bool = False,
    fix: bool = True,
    realization_list: list[str] = ['r1'],
    reader_kwargs: dict = None,
    startdate = None,
    enddate = None,
    freq: str = None,
    loglevel: str = 'WARNING',
):
    """
    Loop over a list of realizations, fetch data using AQUA Reader, and concatenate.

    Args:
        variable (str, optional): Name of the variable to retrieve. Defaults to None.
        ens_dim (str, optional): Dimension name for ensembles (unused directly in concatenation here). Defaults to "ensemble".
        catalog (str, optional): AQUA catalog name. Defaults to None.
        model (str, optional): Model name. Defaults to None.
        exp (str, optional): Experiment name. Defaults to None.
        source (str, optional): Source name. Defaults to None.
        areas (bool, optional): Area extraction flag. Defaults to False.
        fix (bool, optional): Fix flag for the reader. Defaults to True.
        realization_list (list[str], optional): List of realizations to load. Defaults to ['r1'].
        reader_kwargs (dict, optional): Additional kwargs for the Reader. Defaults to None.
        startdate (str, optional): Start date string. Defaults to None.
        enddate (str, optional): End date string. Defaults to None.
        freq (str, optional): Frequency parameter for the Reader. Defaults to None.
        loglevel (str, optional): Logging level. Defaults to 'WARNING'.

    Returns:
        xarray.Dataset: Dataset containing all requested realizations for a given 
        model concatenated along the `model` dimension, with a new `realization` coordinate.
    """
    logger = log_configure(log_name="loop_over_realizations", log_level=loglevel)
    logger.info("Looping over realizations to be merged")

    #Initialize an empty list
    ds_list = []

    if isinstance(realization_list, str):
        realization_list = [realization_list]
    
    if realization_list is None: realization_list = ["r1"]
    for r in realization_list:
        reader = Reader(
            catalog=catalog,
            model=model,
            exp=exp,
            source=source,
            realization=r,
            areas=areas,
            reader_kwargs=reader_kwargs,
            startdate=startdate,
            enddate=enddate,
            fix=fix,
            freq=freq,
            loglevel=loglevel,
        )
        ds = reader.retrieve(var=variable)
        #ds = ds[variable]

        # Add a coordinate to know which realization is this
        ds = ds.assign_coords(realization=r)
        ds_list.append(ds)

    return xr.concat(ds_list, dim=model)

def merge_from_data_files(
    variable: str = None,
    ens_dim: str = "ensemble",
    model_names: list[str] = None,
    data_path_list: list[str] = None,
    # region: str = None,
    # lon_limits: list[float] = None,
    # lat_limits: list[float] = None,
    startdate: str = None,
    enddate: str = None,
    loglevel: str = "WARNING",
):
    """
    Merge ensemble NetCDF files along the ensemble dimension with optional temporal selection.

    This function loads NetCDF files from the given paths via Dask auto-chunking, assigns 
    an ensemble dimension, optionally subsets the data by start and end dates, and concatenates 
    the datasets into a single xarray.Dataset along `ens_dim`. An inner join is used to 
    automatically align the datasets and keep only overlapping time steps. Model names are 
    assigned to each ensemble member for metadata tracking.

    Args:
        variable (str, optional): Name of the variable to merge. Defaults to None.
        ens_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".
        model_names (list[str], optional): List of model names corresponding to the sequence 
            of files in `data_path_list`. If multiple realizations exist for a model, repeat 
            model names accordingly.
        data_path_list (list[str], optional): List of file paths to NetCDF datasets. Mandatory.
        startdate (str, optional): Start date for temporal subsetting (YYYY-MM-DD). Defaults to None.
        enddate (str, optional): End date for temporal subsetting (YYYY-MM-DD). Defaults to None.
        loglevel (str, optional): Logging level. Defaults to "WARNING".

    Returns:
        xarray.Dataset: Merged dataset concatenated along `ens_dim`, with model names in metadata.
        If the dataset has a time dimension, the data is sliced according to startdate and enddate.

    TODO:
        - Add support for spatial subsetting via `region`, `lon_limits`, and `lat_limits`.
        - Handle datasets with multiple variables more flexibly.
        - Include additional metadata about sources or experiments if available.
    """

    logger = log_configure(log_name="merge_from_data_files", log_level=loglevel)
    logger.info("Loading and merging the ensemble dataset")

    if data_path_list is None or not data_path_list:
        raise ValueError("data_path_list must be provided and cannot be empty.")

    # Handle Model Names
    if model_names is not None:
        model_counts = dict(Counter(model_names))
    if model_names is None or len(model_counts.keys()) <= 1:
        logger.info("Single model ensemble members are given")
        if model_names is None:
            logger.info("No model name is given. Assigning a default model_name.")
            model_names = ["model_name"] * len(data_path_list)
    else:
        logger.info("Multi-model ensemble members are given")

    # Preprocessing Function to Filter Variables efficiently
    def preprocess_keep_var(ds):
        if variable is not None and variable in ds.data_vars:
            return ds[[variable]]
        return ds

    # Optimized Lazy Loading & Merging
    ens_dataset = xr.open_mfdataset(
        data_path_list,
        chunks="auto",
        parallel=True,
        combine="nested",
        concat_dim=ens_dim,
        preprocess=preprocess_keep_var,
        
        # 'join="inner"' automatically aligns the datasets and keeps only the 
        # overlapping time steps. This completely replaces your manual 
        # tmp_min_date_list and tmp_max_date_list overlap calculations!
        join="inner" 
    )

    # Assign Coordinates (Ensemble Names and Models)
    # Generates ['r1', 'r2', 'r3' ...] based on the number of files
    ensemble_names = [f"r{i}" for i in range(1, len(data_path_list) + 1)]
    
    ens_dataset = ens_dataset.assign_coords({
        ens_dim: ensemble_names,           # Assigns r1, r2, etc. to the ensemble dim
        "model": (ens_dim, model_names)    # Attaches model names to the ensemble dim
    })

    # Time Slicing
    if "time" in ens_dataset.dims:
        if startdate is not None and enddate is not None:
            ens_dataset = ens_dataset.sel(time=slice(startdate, enddate))
        logger.info("Finished loading the ensemble timeseries datasets")
    else:
        logger.info("Finished loading the ensemble datasets (no time dimension)")

    # Apply Metadata Attributes
    ens_dataset.attrs["description"] = f"Dataset merged along {ens_dim} for ensemble statistics"
    ens_dataset.attrs["model"] = model_names

    return ens_dataset

def compute_statistics(variable: str = None, ds: xr.Dataset = None, ens_dim: str = "ensemble", loglevel="WARNING"):
    """
    Compute mean and standard deviation (POINT-WISE for timeseries) for single- and multi-model ensembles.

    - Single-model: computes unweighted mean and standard deviation along `ens_dim`.
    - Multi-model: computes weighted mean and standard deviation based on the number
      of realizations per model.

    Args:
        variable (str): Name of the variable to compute statistics for.
        ds (xr.Dataset): xarray.Dataset containing ensemble data along `ens_dim`.
        ens_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".
        loglevel (str, optional): Logging level. Defaults to "WARNING".

    Returns:
        tuple:
            Single-model: (ds_mean, ds_std)
            Multi-model: (weighted_mean, weighted_std)

    Raises:
        NoDataError: If `ds` is None.

    Notes:
        - Point-wise STD for ensemble timeseries.
        - The function detects multi-model ensembles via the presence of a 'model' coordinate.
        - Weighted statistics normalize contributions by the number of realizations per model.
        - Attributes 'description' are added to weighted statistics for clarity.

    TODO:
        - Add support for additional statistics (median, percentile).
        - Allow optional masking of NaN values across models before computing statistics.
        - Optimize memory usage for very large ensembles.
        - Include option to return a combined dataset with both mean and std.
    """ 
    logger = log_configure(log_name="compute_statistics", log_level=loglevel)
    logger.info("Computing statistics of the ensemble dataset")

    if ds is None:
        raise NoDataError("No data is given to compute_statistics")

    # Case 1: dataset has 'model' coordinate
    if "model" in ds.coords:
        unique_models = np.unique(ds["model"].values)
        if len(unique_models) <= 1:
            logger.info("Single-model ensemble detected")
            # unweighted mean and std
            ds_mean = ds[variable].mean(dim=ens_dim, skipna=False, keep_attrs=True)
            ds_std = ds[variable].std(dim=ens_dim, skipna=False, keep_attrs=True)
            return ds_mean, ds_std
        else:
            logger.info("Multi-model ensemble detected")
            # Weighted mean/std based on realizations
            # Step 1: compute number of realizations per model in the dataset
            model_counts = {model: np.sum(ds["model"].values == model) for model in unique_models}

            # Step 2: assign weight for each ensemble member
            weights = xr.DataArray([model_counts[m] for m in ds["model"].values], dims=ens_dim, coords={ens_dim: ds[ens_dim]})

            # Step 3: normalize weights
            normalized_weights = weights / weights.sum()

            # Step 4: compute weighted mean
            weighted_mean = (ds[variable] * normalized_weights).sum(dim=ens_dim, skipna=False, keep_attrs=True)

            # Step 5: compute weighted std
            broadcast_mean = weighted_mean.expand_dims({ens_dim: ds.dims[ens_dim]}).transpose(*ds[variable].dims)
            weighted_var = (((ds[variable] - broadcast_mean) ** 2) * normalized_weights).sum(
                dim=ens_dim, skipna=False, keep_attrs=True
            )
            weighted_std = np.sqrt(weighted_var)

            weighted_mean.attrs.update(
                {
                    "description": "Weighted mean based on actual model realizations",
                }
            )
            weighted_std.attrs.update(
                {
                    "description": "Weighted std based on actual model realizations",
                }
            )

            return weighted_mean, weighted_std

    else:
        # Case 2: no model coordinate, assume single-model ensemble
        logger.info("Single-model ensemble detected (no 'model' coordinate)")
        ds_mean = ds[variable].mean(dim=ens_dim, skipna=False, keep_attrs=True)
        ds_std = ds[variable].std(dim=ens_dim, skipna=False, keep_attrs=True)
        return ds_mean, ds_std

def center_timestamp(time: pd.Timestamp, freq: str):
    """
    Center the time value at the center of the month or year

    Args:
        time (str): The time value
        freq (str): The frequency of the time period (only 'monthly' or 'annual')

    Returns:
        pd.Timestamp: The centered time value

    Raises:
        ValueError: If the frequency is not supported
    """
    if freq == "monthly":
        center_time = time + pd.DateOffset(days=15)
    elif freq == "annual":
        center_time = time + pd.DateOffset(months=6)
    else:
        raise ValueError(f"Frequency {freq} not supported")

    return center_time

def extract_realizations(catalog, model, exp, source):
    """Extract the realizations available for a given catalog, model, exp and source.

    Args:
        catalog (str): Intake catalog name.
        model (str): Model name.
        exp (str): Experiment name.
        source (str): Source name.

    Returns:
        list: List of available realizations.
    """
    configurer = ConfigPath(catalog=catalog, loglevel="WARNING")
    cat, catalog_file, machine_file = configurer.deliver_intake_catalog(catalog=catalog, model=model, exp=exp, source=source)

    expcat = cat()[model][exp]
    esmcat = expcat[source].describe().get("user_parameters", {})

    for parameter in esmcat:
        name = parameter.get("name")

        if name == "realization":
            realization = parameter.get("allowed")
            return realization
    return None

def extract_realizations_list(catalog, model, exp, source, loglevel="WARNING"):
    """
    Extract the available realizations accessing the uninstantiated catalog directly.

    Args:
        catalog (str): Intake catalog name.
        model (str): Model name.
        exp (str): Experiment name.
        source (str): Source name.
        loglevel (str, optional): Logging level. Defaults to "WARNING".

    Returns:
        list or None: List of available realizations, or None if the source 
        does not exist or no realization parameter is found.
    """
    logger = log_configure(log_name="extract_realizations_list", log_level=loglevel)
    logger.info("extracting realizations list")

    configurer = ConfigPath(catalog=catalog, loglevel=loglevel)
    cat, catalog_file, machine_file = configurer.deliver_intake_catalog(catalog=catalog, model=model, exp=exp, source=source)

    expcat = cat()[model][exp]
    
    # Access the uninstantiated catalog entry using `_entries`
    if source not in expcat._entries:
        logger.warning(f"No realization(s) found!") 
        return None
        
    entry = expcat._entries[source]
    
    # describe() on the entry returns a dictionary containing 'user_parameters'
    user_parameters = entry.describe().get("user_parameters", [])

    # user_parameters is parsed by Intake into a list of dictionaries
    for parameter in user_parameters:
        name = parameter.get("name")
        if name == "realization":
            realizations = parameter.get("allowed")
            logger.info(f"Realizations found in the catalog: {realizations}") 
            return realizations
    return 

def generate_realizations_path(catalog: str, model: str, exp: str, diagnostic_name: str, diagnostic_product: str, variable: str, file_dir: str, realization_list: list[str] = None,  extra_keys=None, file_format: str =".nc", loglevel="WARNING"):
    """
    Generate output file paths for specific diagnostic realizations.

    Leverages the `OutputSaver` class to standardly generate filenames 
    and concatenates them with the specified output directory and file extension.

    Args:
        catalog (str): Name of the data catalog.
        model (str): Name of the model.
        exp (str): Name of the experiment.
        diagnostic_name (str): The name of the diagnostic being run.
        diagnostic_product (str): The specific product/output type of the diagnostic.
        variable (str): The variable associated with the diagnostic.
        file_dir (str): Base directory where the files will be saved.
        realization_list (list[str], optional): List of ensemble realizations. 
            If None, generates a single path without realization info. Defaults to None.
        extra_keys (dict, optional): Additional keys for the `OutputSaver` filename generation. Defaults to None.
        file_format (str, optional): Extension for the output files. Defaults to ".nc".
        loglevel (str, optional): Logging level. Defaults to "WARNING".

    Returns:
        list[str]: A list of full file paths correctly formatted for the diagnostic outputs.
    """
    logger = log_configure(log_name="generate_realizations_path", log_level=loglevel)
    logger.info("Generating realization paths")
    
    filenames = []
    if realization_list:
        for r in realization_list:
            outputsaver = OutputSaver(diagnostic=diagnostic_name, catalog=catalog, model=model, exp=exp, realization=r, outputdir=file_dir, loglevel=loglevel)
            _path = outputsaver.generate_name(diagnostic_product=diagnostic_product, extra_keys=extra_keys)
            path = file_dir + "/" + _path + file_format 
            filenames.append(path)
    else:
        outputsaver = OutputSaver(diagnostic=diagnostic_name, catalog=catalog, model=model, exp=exp, outputdir=file_dir, loglevel=loglevel)
        _path = outputsaver.generate_name(diagnostic_product=diagnostic_product, extra_keys=extra_keys)
        path = file_dir + "/" + _path + file_format 
        filenames.append(path)

    logger.debug(f"generated file names for realizations are {filenames}")
    return filenames
        
 
