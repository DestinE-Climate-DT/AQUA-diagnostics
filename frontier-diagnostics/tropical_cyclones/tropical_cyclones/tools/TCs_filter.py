# TCs_filter.py
#
# SUBTROPICAL JET FILTER FOR TROPICAL CYCLONES
# =============================================
#
# This module implements the Subtropical Jet (STJ) filtering methodology
# adapted from Bourdin et al. (2022) to distinguish tropical from extratropical
# cyclones based on their position relative to the jet stream.
#
# METHODOLOGY:
# ------------
# The filter uses ERA5 reanalysis data at 250 hPa to:
# 1. Compute monthly climatology of zonal wind (U) and total wind speed (V)
# 2. Identify the subtropical jet latitude as the poleward extent where:
#    - U >= 25 m/s (zonal wind threshold)
#    - V >= 25 m/s (total wind speed threshold)
# 3. Classify each TC observation as "tropical" if it is located at least
#    8° equatorward of the jet latitude for that month
# 4. Retain only TC tracks that have at least 1 tropical timestep
#
# The buffer of 8° (default) provides a conservative separation between
# tropical systems (warm-core, low-latitude) and extratropical transitions
# (baroclinic, jet-influenced systems).
#
# FUNCTIONS:
# ----------
# filter_STJ_Bourdin250_monthly():
#   Main filtering function - reads trajectory file, applies STJ filter,
#   outputs filtered file with suffix "_filtered_STJ_Bourdin250_monthly.txt"
#
# Helper functions for ERA5 processing:
#   - load_era5_250(): Load 250 hPa U/V wind components
#   - compute_monthly_climatology_250(): Multi-year monthly climatology
#   - compute_jet_latitudes(): Extract jet position for each month
#
# REQUIREMENTS:
# -------------
# - xarray, numpy
# - ERA5 data at 250 hPa (6-hourly, separate U/V files per year)
# - TC trajectory file in standard format (track_id, year, month, ..., lat, ...)
#
# ===============================================================================

import xarray as xr
import numpy as np
import os


# ============================================================================
# ERA5 DATA LOADING
# ============================================================================

def load_era5_250(year, era5_dir):
    """
    Load ERA5 250 hPa U and V wind components for a given year.
    
    Args:
        year: Year to load (int)
        era5_dir: Directory containing ERA5 netCDF files
    
    Returns:
        ds_u, ds_v: xarray Datasets with U and V components
    """
    fu = f"{era5_dir}/ERA5_u_component_of_wind_250hPa_6hr_{year}.nc"
    fv = f"{era5_dir}/ERA5_v_component_of_wind_250hPa_6hr_{year}.nc"

    ds_u = xr.open_dataset(fu)
    ds_v = xr.open_dataset(fv)

    # Standardize time coordinate name
    if "valid_time" in ds_u.coords:
        ds_u = ds_u.rename({"valid_time": "time"})
    if "valid_time" in ds_v.coords:
        ds_v = ds_v.rename({"valid_time": "time"})

    # Convert longitude to [-180, 180]
    if ds_u.longitude.max() > 180:
        new_lon = ((ds_u.longitude + 180) % 360) - 180
        ds_u = ds_u.assign_coords(longitude=new_lon)
        ds_v = ds_v.assign_coords(longitude=new_lon)

    # Remove pressure level dimension if present
    if "pressure_level" in ds_u.dims:
        ds_u = ds_u.squeeze("pressure_level")
    if "pressure_level" in ds_v.dims:
        ds_v = ds_v.squeeze("pressure_level")

    return ds_u, ds_v


# ============================================================================
# CLIMATOLOGY COMPUTATION
# ============================================================================

def compute_monthly_climatology_250(years, era5_dir):
    """
    Compute monthly climatology of U, V, and total wind speed at 250 hPa.
    
    Args:
        years: List of years to include in climatology
        era5_dir: Directory containing ERA5 files
    
    Returns:
        u_mon: Monthly mean zonal wind [12 months x lat x lon]
        v_mon: Monthly mean meridional wind [12 months x lat x lon]
        V_mon: Monthly mean total wind speed [12 months x lat x lon]
    """
    u_list = []
    v_list = []

    for y in years:
        print(f"  Loading {y} ERA5 250 hPa...")
        ds_u, ds_v = load_era5_250(y, era5_dir)
        u_list.append(ds_u["u"])
        v_list.append(ds_v["v"])

    # Concatenate all years
    u_all = xr.concat(u_list, dim="time")
    v_all = xr.concat(v_list, dim="time")

    # Monthly climatology
    u_mon = u_all.groupby("time.month").mean("time")
    v_mon = v_all.groupby("time.month").mean("time")

    # Total wind speed
    V_mon = np.sqrt(u_mon**2 + v_mon**2)

    return u_mon, v_mon, V_mon


# ============================================================================
# JET LATITUDE DETECTION
# ============================================================================

def compute_jet_latitudes(u_mon, V_mon,
                          lat_min=10, lat_max=50,
                          u_thresh=25, V_thresh=25):
    """
    Extract subtropical jet latitude for each month using Bourdin method.
    
    The jet latitude is defined as the poleward extent where zonally-averaged
    winds exceed both U and V thresholds within the latitude band [lat_min, lat_max].
    
    Args:
        u_mon: Monthly climatology of zonal wind
        V_mon: Monthly climatology of total wind speed
        lat_min: Equatorward boundary for jet search (degrees)
        lat_max: Poleward boundary for jet search (degrees)
        u_thresh: Minimum zonal wind threshold (m/s)
        V_thresh: Minimum total wind threshold (m/s)
    
    Returns:
        jet_lat_N: Jet latitude for each month in NH [12 values]
        jet_lat_S: Jet latitude for each month in SH [12 values]
    """
    lat = u_mon.latitude.values
    jet_lat_N = np.full(12, np.nan)
    jet_lat_S = np.full(12, np.nan)

    for m in range(12):
        # Zonal mean profiles
        u_prof = u_mon.isel(month=m).mean(dim="longitude").values
        V_prof = V_mon.isel(month=m).mean(dim="longitude").values

        # Apply thresholds (simplified Bourdin criterion)
        mask = (u_prof >= u_thresh) & (V_prof >= V_thresh)

        # Northern Hemisphere: find maximum latitude meeting criteria
        idxN = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        validN = idxN[mask[idxN]] if len(idxN) > 0 else []
        if len(validN) > 0:
            jet_lat_N[m] = lat[validN].max()

        # Southern Hemisphere: find minimum (most equatorward) latitude
        idxS = np.where((lat <= -lat_min) & (lat >= -lat_max))[0]
        validS = idxS[mask[idxS]] if len(idxS) > 0 else []
        if len(validS) > 0:
            jet_lat_S[m] = lat[validS].min()

    return jet_lat_N, jet_lat_S


# ============================================================================
# TRAJECTORY FILE PARSING HELPER
# ============================================================================

def parse_track_line(line):
    """
    Parse a trajectory file line supporting both space and comma separators.
    
    Args:
        line: String line from trajectory file
    
    Returns:
        List of fields, or None if line is invalid/comment
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # Try comma-separated first
    if ',' in line:
        parts = [p.strip() for p in line.split(',')]
    else:
        parts = line.split()
    
    # Remove trailing commas from values
    parts = [p.rstrip(',') for p in parts]
    
    return parts if len(parts) >= 11 else None


# ============================================================================
# MAIN FILTER FUNCTION
# ============================================================================

def filter_STJ_Bourdin250_monthly(trajfile, era5_dir,
                                  buffer_deg=8.0,
                                  lat_min=10, lat_max=50,
                                  u_thresh=25, V_thresh=25):
    """
    Apply Subtropical Jet filter to TC trajectory file.
    
    This function:
    1. Reads all TC trajectories from the input file
    2. Computes ERA5 250 hPa monthly climatology for the data period
    3. Identifies subtropical jet position for each month (NH and SH)
    4. Classifies each TC observation as tropical if sufficiently equatorward
       of the jet (>= buffer_deg degrees)
    5. Retains only tracks with at least 1 tropical timestep
    6. Writes filtered trajectories to new file
    
    Args:
        trajfile: Path to input trajectory file
        era5_dir: Directory containing ERA5 250 hPa data
        buffer_deg: Buffer zone equatorward of jet (default 8°)
        lat_min: Equatorward boundary for jet search (default 10°)
        lat_max: Poleward boundary for jet search (default 50°)
        u_thresh: Zonal wind threshold for jet detection (default 25 m/s)
        V_thresh: Total wind threshold for jet detection (default 25 m/s)
    
    Returns:
        out_file: Path to filtered output file
        jet_lat_N: NH jet latitudes [12 months]
        jet_lat_S: SH jet latitudes [12 months]
    """
    print(f"[INFO] STJ filter (Bourdin 250 hPa monthly) applied to {trajfile}")
    print(f"\n{'='*70}")
    print(f"STJ FILTER - Bourdin et al. (2022) method")
    print(f"{'='*70}")
    print(f"Input file: {trajfile}")
    print(f"Buffer zone: {buffer_deg}° equatorward of jet")
    print(f"Jet detection: U>={u_thresh} m/s, V>={V_thresh} m/s")
    
    # ========================================================================
    # STEP 1: Determine years in trajectory file
    # ========================================================================
    years = set()
    with open(trajfile, "r") as f:
        for line_num, line in enumerate(f):
            # Skip header (first line)
            if line_num == 0:
                continue
            
            parts = parse_track_line(line)
            if parts is None:
                continue
            
            try:
                # Year is in column 1 (0-indexed)
                year = int(parts[1])
                # Convert 2-digit year to 4-digit (1950-2049)
                if year < 100:
                    year = 1900 + year if year >= 50 else 2000 + year
                years.add(year)
            except (ValueError, IndexError):
                continue
    
    years = sorted(list(years))
    print(f"Data period: {years[0]}–{years[-1]} ({len(years)} years)")
    
    # ========================================================================
    # STEP 2: Compute ERA5 climatology
    # ========================================================================
    print(f"\nComputing ERA5 250 hPa monthly climatology...")
    u_mon, v_mon, V_mon = compute_monthly_climatology_250(years, era5_dir)
    
    # ========================================================================
    # STEP 3: Extract jet latitudes
    # ========================================================================
    print(f"\nDetecting subtropical jet positions...")
    jet_lat_N, jet_lat_S = compute_jet_latitudes(
        u_mon, V_mon,
        lat_min=lat_min,
        lat_max=lat_max,
        u_thresh=u_thresh,
        V_thresh=V_thresh
    )



    print(f"\nNH Jet latitudes (°N):")
    for m in range(12):
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m]
        val = jet_lat_N[m]
        print(f"  {month_name}: {val:.2f}" if not np.isnan(val) else f"  {month_name}: N/A")
    
    print(f"\nSH Jet latitudes (°S):")
    for m in range(12):
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m]
        val = jet_lat_S[m]
        print(f"  {month_name}: {val:.2f}" if not np.isnan(val) else f"  {month_name}: N/A")
    
    print("[INFO] Jet latitudes NH:", np.round(jet_lat_N, 2))
    print("[INFO] Jet latitudes SH:", np.round(jet_lat_S, 2))
    # ========================================================================
    # STEP 4: Apply filter to trajectories
    # ========================================================================
    print(f"\nApplying STJ filter to trajectories...")
    
    # Output file name
    out_file = trajfile.replace(".txt", "_filtered_STJ_Bourdin250_monthly.txt")
    out_file = out_file.replace(".csv", "_filtered_STJ_Bourdin250_monthly.txt")
    
    fout = open(out_file, "w")
    
    # Track-by-track filtering
    keep_lines = []
    tropical_counter = 0
    total_tracks = 0
    kept_tracks = 0
    
    def flush_track():
        """Write accumulated track to output if it has >= 1 tropical timestep"""
        nonlocal tropical_counter, keep_lines, kept_tracks
        if tropical_counter >= 1:
            for L in keep_lines:
                fout.write(L)
            kept_tracks += 1
        tropical_counter = 0
        keep_lines = []
    
    # Parse trajectory file
    current_storm = None
    with open(trajfile, "r") as fin:
        for line_num, line in enumerate(fin):
            # Skip header (first line)
            if line_num == 0:
                continue
            
            parts = parse_track_line(line)
            if parts is None:
                continue
            
            try:
                storm_id = parts[0]
                year = int(parts[1])
                # Convert 2-digit year
                if year < 100:
                    year = 1900 + year if year >= 50 else 2000 + year
                month = int(parts[2])
                lat = float(parts[8])
                
                # Check if new storm
                if storm_id != current_storm:
                    if current_storm is not None:
                        total_tracks += 1
                    flush_track()
                    current_storm = storm_id
                
                mi = month - 1  # 0-indexed month
                
                # Classify as tropical or extratropical
                if lat >= 0:  # Northern Hemisphere
                    jet = jet_lat_N[mi]
                    is_trop = (np.isnan(jet) or lat <= jet - buffer_deg)
                else:  # Southern Hemisphere
                    jet = jet_lat_S[mi]
                    is_trop = (np.isnan(jet) or lat >= jet + buffer_deg)
                
                if is_trop:
                    tropical_counter += 1
                
                keep_lines.append(line)
                
            except (ValueError, IndexError) as e:
                print(f"[WARNING] Skipping malformed line: {line.strip()[:50]}...")
                continue
    
    # Flush last track
    if current_storm is not None:
        total_tracks += 1
    flush_track()
    
    fout.close()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"FILTERING COMPLETE")
    print(f"{'='*70}")
    print(f"Total tracks processed: {total_tracks}")
    print(f"Tracks retained: {kept_tracks} ({100*kept_tracks/max(total_tracks,1):.1f}%)")
    print(f"Tracks removed: {total_tracks - kept_tracks}")
    print(f"\nOutput file: {out_file}")
    print(f"{'='*70}\n")
    
    return out_file, jet_lat_N, jet_lat_S
    print(f"[INFO] Filter complete. Output:", out_file)
