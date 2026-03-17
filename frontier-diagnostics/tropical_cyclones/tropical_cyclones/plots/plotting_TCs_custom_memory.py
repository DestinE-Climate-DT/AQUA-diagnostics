# plotting_TCs_custom_memory.py
# 
# TROPICAL CYCLONE VISUALIZATION - MEMORY-OPTIMIZED VERSION
# ==========================================================
#
# This module provides plotting functions for TC trajectories that accept
# BOTH file paths (backward compatible) AND TCDataManager objects (memory-
# optimized, 100-1000x faster for multiple plots).
#
# All functions save outputs in both PDF and NetCDF formats where applicable.
#
# MAIN FUNCTIONS:
# ---------------
# 1. plot_trajectories_direct(data_or_file, tdict, max_timesteps)
#    Simple scatter plot of all TC tracks (black dots). Quick diagnostic view
#    showing spatial distribution. Saves PDF + NetCDF with point coordinates.
#
# 2. plot_trajectories_colored(data_or_file, tdict, color_by, category, max_timesteps)
#    TC tracks as continuous lines colored by Saffir-Simpson category. Can plot
#    ALL storms (category=None) or filter by specific category (0-5 for TD through
#    Cat 5). Storms sorted by intensity for optimal visualization. Handles dateline
#    crossings. Saves PDF + NetCDF with trajectory data and categories.
#
# 3. plot_density_scatter(data_or_file, tdict, max_timesteps, sample_size)
#    KDE-based density scatter where each point is colored by local track density
#    (normalized 0-1). Points sorted by density for better visualization. Useful
#    for identifying TC activity hotspots. Saves PDF only.
#
# 4. plot_track_density_grid(data_or_file, tdict, grid_size, max_timesteps, category)
#    Gridded density map showing "transits per month" following HighResMIP-PRIMAVERA
#    standards. Discrete logarithmic colorbar with adaptive boundaries. Can plot ALL
#    TCs (category=None) or filter by category (0-5) to show spatial distribution of
#    specific intensity ranges (e.g., category=3 for major hurricanes only). Saves
#    PDF + NetCDF with gridded density fields and metadata.
#
# 5. plot_density_scatter_by_category(data_or_file, tdict, max_timesteps, sample_size)
#    6-panel subplot showing KDE density for each Saffir-Simpson category (TD through
#    Cat 5). Each panel uses category-specific colormap. Enables comparison of spatial
#    patterns across intensity ranges. Saves PDF only.
#
# 6. plot_tc_duration_distribution(data_or_file, tdict)
#    Histogram + KDE showing distribution of TC lifetimes with mean and median reference
#    lines. Computes duration statistics (mean, median, std, min, max) for all storms.
#    Saves PDF only.
#
# 7. plot_tc_duration_by_category(data_or_file, tdict)
#    Normalized histograms (PDFs) of TC durations separated by Saffir-Simpson category.
#    All curves normalized for direct comparison regardless of sample size. Shows how
#    duration varies with intensity. Saves PDF only.
#
# 8. plot_tc_basin_doughnut(data_or_file, tdict, reference_freq)
#    Doughnut chart showing annual TC frequency for each ocean basin following Roberts
#    et al. (2020) methodology. Includes NH/SH totals, seasonal filtering (NH: May-Nov,
#    SH: Oct-May), and comparison bar chart with literature values. Uses IBTrACS basin
#    definitions. Saves two PDFs (doughnut + comparison chart).
#
# USAGE:
# ------
# Option 1 (traditional, slow - reads from disk each time):
#   plot_trajectories_direct("filtered_file.txt", config)
#
# Option 2 (memory-optimized, fast - loads once, reuses):
#   from tc_data_manager import TCDataManager
#   tc_data = TCDataManager("filtered_file.txt")  # Load once
#   plot_trajectories_direct(tc_data, config)      # Instant!
#   plot_density_scatter(tc_data, config)          # Instant!
#   plot_track_density_grid(tc_data, config)       # Instant!
#
# All functions produce publication-ready outputs with Cartopy PlateCarree
# projection and standardized naming conventions.
# 
# All plotting functions accept an optional cat_method argument:
#
#   cat_method='slp'   (default) — Saffir-Simpson category derived from
#                                   minimum SLP using category_from_slp_pa().
#                                   Works with any input file (ERA5, IBTrACS, models).
#
#   cat_method='sshs'            — Official IBTrACS Saffir-Simpson category
#                                   read directly from column 11 (usa_sshs,
#                                   wind-based, NOAA definition).
#                                   Only meaningful for IBTrACS-converted files.
#                                   Falls back to 'slp' if sshs not available.
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import os
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap
from scipy.stats import gaussian_kde
from datetime import datetime
import xarray as xr
 
 
# ===============================================================================
# CATEGORY HELPERS
# ===============================================================================
 
def category_from_slp_pa(slp_pa):
    """Saffir-Simpson category from SLP in Pascals (0=TD/TS, 1-5=Cat1-5)."""
    slp_hpa = slp_pa / 100.0
    if slp_hpa >= 1005:
        return 0
    elif slp_hpa >= 990:
        return 1
    elif slp_hpa >= 975:
        return 2
    elif slp_hpa >= 960:
        return 3
    elif slp_hpa >= 945:
        return 4
    else:
        return 5
 
 
def _get_cat_array(storm_data, method='slp'):
    """
    Return per-timestep category array for a single storm.
 
    Args:
        storm_data : dict with keys 'slp', and optionally 'sshs'
        method     : 'slp' → derived from pressure
                     'sshs' → read from IBTrACS usa_sshs column (fallback to slp)
 
    Returns:
        np.ndarray of int, shape (n_timesteps,), values 0-5
    """
    if method == 'sshs' and 'sshs' in storm_data:
        raw = np.array(storm_data['sshs'], dtype=int)
        # IBTrACS sshs: -15=missing, -5..−1=non-TC, 0=TS, 1-5=Cat1-5
        # Remap: anything < 0 → 0 (treat TD/TS/disturbance as category 0)
        cats = np.where(raw < 0, 0, raw)
        cats = np.clip(cats, 0, 5)
        return cats
    # default / fallback
    return np.array([category_from_slp_pa(s) for s in storm_data['slp']])
 
 
def _peak_category(storm_data, method='slp'):
    """Return the peak (maximum) category reached by a storm."""
    return int(np.max(_get_cat_array(storm_data, method)))
 
 
# ===============================================================================
# LABELS AND COLOURS (shared across functions)
# ===============================================================================
 
_CAT_NAMES_SLP = [
    'TD (≥1005 hPa)', 'Cat 1 (990–1004 hPa)', 'Cat 2 (975–989 hPa)',
    'Cat 3 (960–974 hPa)', 'Cat 4 (945–959 hPa)', 'Cat 5 (<945 hPa)'
]
 
_CAT_NAMES_SSHS = [
    'TD/TS (sshs ≤ 0)', 'Cat 1 (sshs=1)', 'Cat 2 (sshs=2)',
    'Cat 3 (sshs=3)',   'Cat 4 (sshs=4)', 'Cat 5 (sshs=5)'
]
 
_CAT_COLORS = ['lightgreen', 'gold', 'orange', 'red', 'darkred', 'purple']
 
 
def _cat_names(method):
    return _CAT_NAMES_SSHS if method == 'sshs' else _CAT_NAMES_SLP
 
 
def _method_suffix(method):
    """Short string appended to output filenames."""
    return '_sshs' if method == 'sshs' else '_slp'
 
 
# ===============================================================================
# I/O HELPERS
# ===============================================================================
 
def get_basin_ibtracs(lon, lat):
    lon_360 = lon if lon >= 0 else lon + 360
    if (260 <= lon_360 <= 360 or 0 <= lon_360 <= 20) and 0 <= lat <= 70:
        return 'North Atlantic'
    if 180 <= lon_360 < 260 and 0 <= lat <= 60:
        return 'East Pacific'
    if 100 <= lon_360 < 180 and 0 <= lat <= 60:
        return 'West Pacific'
    if 30 <= lon_360 < 100 and 0 <= lat <= 40:
        return 'North Indian'
    if 20 <= lon_360 < 135 and -40 <= lat < 0:
        return 'South Indian'
    if (135 <= lon_360 <= 360 or 0 <= lon_360 < 240) and -40 <= lat < 0:
        return 'South Pacific'
    if (290 <= lon_360 <= 360 or 0 <= lon_360 <= 20) and -40 <= lat < 0:
        return 'South Atlantic'
    return 'Other'
 
 
def getTrajectories_direct(filename):
    """
    Read trajectories from the 12-column ASCII file.
 
    Column layout:
      [0]  storm_id
      [1]  year   [2] month  [3] day  [4] hour  [5] minute  [6] step
      [7]  lon    [8] lat
      [9]  slp_Pa  [10] wind_kts
      [11] sshs (int, IBTrACS usa_sshs) OR basin string (ERA5/TE files)
 
    Column 11 is read as sshs only when it parses as an integer;
    otherwise it is stored as 'basin' and sshs is left absent.
 
    Returns:
        numtraj    : int
        maxNumPts  : int
        trajectories : list of storm dicts
    """
    print(f"Reading trajectories from file: {filename}")
 
    trajectories_dict = {}
 
    with open(filename, 'r') as f:
        for line in f:
            if 'track_id' in line or 'year' in line:
                continue
 
            parts = line.strip().split()
            if len(parts) != 12:
                continue
 
            storm_id = parts[0]
            year     = int(parts[1])
            month    = int(parts[2])
            day      = int(parts[3])
            hour     = int(parts[4])
            lon      = float(parts[7])
            lat      = float(parts[8])
            slp      = float(parts[9])
            wind     = float(parts[10])
 
            # Column 11: sshs (IBTrACS) or basin string (ERA5/TE)
            try:
                sshs_val = int(parts[11])
                col11_is_sshs = True
            except ValueError:
                sshs_val = None
                col11_is_sshs = False
 
            if storm_id not in trajectories_dict:
                trajectories_dict[storm_id] = {
                    'id':    storm_id,
                    'lon':   [], 'lat':  [],
                    'slp':   [], 'wind': [],
                    'year':  [], 'month': [],
                    'day':   [], 'hour':  [],
                }
                if col11_is_sshs:
                    trajectories_dict[storm_id]['sshs'] = []
 
            trajectories_dict[storm_id]['lon'].append(lon)
            trajectories_dict[storm_id]['lat'].append(lat)
            trajectories_dict[storm_id]['slp'].append(slp)
            trajectories_dict[storm_id]['wind'].append(wind)
            trajectories_dict[storm_id]['year'].append(year)
            trajectories_dict[storm_id]['month'].append(month)
            trajectories_dict[storm_id]['day'].append(day)
            trajectories_dict[storm_id]['hour'].append(hour)
            if col11_is_sshs and 'sshs' in trajectories_dict[storm_id]:
                trajectories_dict[storm_id]['sshs'].append(sshs_val)
 
    # Convert lists → numpy arrays
    trajectories_list = []
    for storm_id in sorted(trajectories_dict.keys(),
                           key=lambda x: int(x) if x.isdigit() else x):
        sd = trajectories_dict[storm_id]
        for key in ['lon', 'lat', 'slp', 'wind', 'year', 'month', 'day', 'hour']:
            sd[key] = np.array(sd[key])
        if 'sshs' in sd:
            sd['sshs'] = np.array(sd['sshs'], dtype=int)
        trajectories_list.append(sd)
 
    numtraj   = len(trajectories_list)
    maxNumPts = max(len(t['lon']) for t in trajectories_list)
 
    has_sshs = all('sshs' in t for t in trajectories_list)
    print(f"Found {numtraj} trajectories  |  sshs column: {'yes' if has_sshs else 'no (SLP fallback)'}")
 
    return numtraj, maxNumPts, trajectories_list
 
 
def _get_data_from_input(data_or_file):
    if hasattr(data_or_file, 'get_all_points'):
        print("Using data from memory (fast)")
        trajectories = data_or_file.get_trajectories()
        data_dict    = data_or_file.get_all_data()
        return trajectories, data_dict
    else:
        print("Reading from file (slow)...")
        _, _, trajectories = getTrajectories_direct(data_or_file)
        return trajectories, None
 
 
# ===============================================================================
# 1. SIMPLE TRACK SCATTER
# ===============================================================================
 
def plot_trajectories_direct(data_or_file, tdict, max_timesteps=None):
    """Simple scatter plot of all TC tracks (black dots). No category needed."""
    trajectories, _ = _get_data_from_input(data_or_file)
    nstorms = len(trajectories)
 
    fig = plt.figure(figsize=(16, 9))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
 
    plt.title(
        f"TC Tracks - {tdict['dataset']['model']} - {tdict['dataset']['exp']}\n"
        f"({nstorms} tracks, {tdict['time']['startdate']} – {tdict['time']['enddate']})",
        fontsize=14, fontweight='bold'
    )
 
    ax.add_feature(cfeature.LAND,      color='lightgrey', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
 
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='k',
                      alpha=0.5, linestyle='--')
    gl.xlabels_top  = False
    gl.ylabels_left = False
    gl.xlocator     = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
 
    all_lons, all_lats, all_ids, all_steps = [], [], [], []
 
    for storm_data in trajectories:
        lon = storm_data['lon']
        lat = storm_data['lat']
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
        lon_plot = np.where(lon > 180, lon - 360, lon)
 
        ax.scatter(lon_plot, lat, color='black', s=22, linewidths=0.5,
                   marker='.', alpha=0.9, transform=ccrs.PlateCarree())
 
        all_lons.extend(lon_plot)
        all_lats.extend(lat)
        all_ids.extend([storm_data['id']] * len(lon_plot))
        all_steps.extend(range(len(lon_plot)))
 
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    sd, ed = tdict['time']['startdate'], tdict['time']['enddate']
    base = f"tracks_{tdict['dataset']['model']}_{tdict['dataset']['exp']}_{sd}_{ed}"
 
    pdf_path = os.path.join(tdict['paths']['plotdir'], f"{base}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=350)
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()
 
    # NetCDF
    try:
        ds = xr.Dataset(
            {'longitude': (['obs'], all_lons),
             'latitude':  (['obs'], all_lats),
             'storm_id':  (['obs'], all_ids),
             'timestep':  (['obs'], all_steps)},
            coords={'obs': np.arange(len(all_lons))},
            attrs={'model': tdict['dataset']['model'],
                   'experiment': tdict['dataset']['exp'],
                   'start_date': sd, 'end_date': ed, 'n_storms': nstorms,
                   'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        nc_path = os.path.join(tdict['paths']['plotdir'], f"{base}.nc")
        ds.to_netcdf(nc_path)
        print(f"✓ NetCDF → {nc_path}")
    except Exception as e:
        print(f"⚠ NetCDF save failed: {e}")
 
 
# ===============================================================================
# 2. COLOURED TRACKS
# ===============================================================================
def plot_trajectories_colored(data_or_file, tdict, color_by='category',
                               category=None, max_timesteps=None,
                               cat_method='slp'):
    """
    Plot TC trajectories colored by Saffir-Simpson category.
    Discrete colour scale (BoundaryNorm, 6 bands).
    Storms sorted weak → strong so Cat5 always on top.

    Args:
        data_or_file : TCDataManager or file path
        tdict        : config dict
        color_by     : kept for API compatibility
        category     : int 0-5 → plot ONLY this category; None → all storms
        max_timesteps: optional limit
        cat_method   : 'slp' (default) or 'sshs' (IBTrACS wind-based)

    Saves: PDF + NetCDF
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    trajectories, _ = _get_data_from_input(data_or_file)
    names  = _cat_names(cat_method)
    suffix = _method_suffix(cat_method)

    # ── discrete colour palette ───────────────────────────────────────────────
    _PALETTE = [
        "#168E90",  # 0 TD/TS  
        "#1EA22D",  # 1 Cat 1  
        "#C2AF06",  # 2 Cat 2  
        "#FD8E2D",  # 3 Cat 3  
        "#FA6BFF",  # 4 Cat 4  
        "#C90D0D",  # 5 Cat 5  rosso
    ]
    cmap   = ListedColormap(_PALETTE)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_d = BoundaryNorm(bounds, cmap.N)

    # ── category filter ───────────────────────────────────────────────────────
    if category is not None:
        cat_names_list = _cat_names(cat_method)
        filtered_trajectories = []
        for storm_data in trajectories:
            peak_cat = _peak_category(storm_data, cat_method)
            if peak_cat == category:
                filtered_trajectories.append(storm_data)
        trajectories = filtered_trajectories
        nstorms = len(trajectories)
        if nstorms == 0:
            print(f"⚠ No trajectories found for {names[category]}")
            return
        print(f"✓ Found {nstorms} trajectories for {names[category]}")
        single_cat = True
    else:
        nstorms    = len(trajectories)
        single_cat = False
        print(f"Plotting all {nstorms} storms (cat_method={cat_method})")

    # ── sort weak → strong so Cat5 plots on top ───────────────────────────────
    trajectories = sorted(trajectories,
                          key=lambda s: _peak_category(s, cat_method))

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      color='lightgrey', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='k',
                      alpha=0.5, linestyle='--')
    gl.xlabels_top  = False
    gl.ylabels_left = False
    gl.xlocator     = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER

    # NetCDF collector
    nc_data = {'storm_id': [], 'longitude': [], 'latitude': [],
               'slp': [], 'category': [], 'timestep': []}

    # ── plot ──────────────────────────────────────────────────────────────────
    for storm_data in trajectories:
        lon = storm_data['lon'].copy()
        lat = storm_data['lat'].copy()
        slp = storm_data['slp'] / 100.0  # Pa → hPa

        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
            slp = slp[:max_timesteps]

        lon_plot = np.where(lon > 180, lon - 360, lon)

        if len(lon_plot) < 2:
            continue

        # one colour per storm based on peak category
        peak_cat = _peak_category(storm_data, cat_method)
        values   = np.full(len(lon_plot) - 1, float(peak_cat))

        # dateline masking
        dlon     = np.abs(np.diff(lon_plot))
        valid    = dlon < 180

        points   = np.array([lon_plot, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segments = segments[valid]
        values   = values[valid]

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm_d,
            linewidth=1.3,
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=10 + peak_cat
        )
        lc.set_array(values)
        ax.add_collection(lc)

        # collect for NetCDF
        cats_ts = _get_cat_array(storm_data, cat_method)
        if max_timesteps is not None:
            cats_ts = cats_ts[:max_timesteps]
        for i in range(len(lon_plot)):
            nc_data['storm_id'].append(storm_data['id'])
            nc_data['longitude'].append(lon_plot[i])
            nc_data['latitude'].append(lat[i])
            nc_data['slp'].append(slp[i])
            nc_data['category'].append(int(cats_ts[i]) if i < len(cats_ts) else 0)
            nc_data['timestep'].append(i)

    # ── colorbar ──────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm_d)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    method_label = 'wind-based (IBTrACS)' if cat_method == 'sshs' else 'SLP-based'
    cbar.set_label(f'Saffir-Simpson Category [{method_label}]', fontsize=12)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(['TD/TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'])

    # ── title ─────────────────────────────────────────────────────────────────
    if single_cat:
        title = (f"TC Tracks – {names[category]}  [{method_label}]\n"
                 f"{tdict['dataset']['model']} – {tdict['dataset']['exp']} "
                 f"({nstorms} storms)")
    else:
        title = (f"TC Tracks coloured by category  [{method_label}]\n"
                 f"{tdict['dataset']['model']} – {tdict['dataset']['exp']} "
                 f"({nstorms} storms)")
    plt.title(title, fontsize=14, fontweight='bold')

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    sd = tdict['time']['startdate'].replace('-', '')
    ed = tdict['time']['enddate'].replace('-', '')
    m  = tdict['dataset']['model'].replace(' ', '_')
    ex = tdict['dataset']['exp'].replace(' ', '_')

    base = (f"tracks_cat{category}{suffix}_{m}_{ex}_{sd}_{ed}"
            if single_cat else
            f"tracks_colored{suffix}_{m}_{ex}_{sd}_{ed}")

    pdf_path = os.path.join(tdict['paths']['plotdir'], f"{base}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=350)
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()

    # ── NetCDF ────────────────────────────────────────────────────────────────
    try:
        ds = xr.Dataset(
            {'longitude': (['obs'], nc_data['longitude']),
             'latitude':  (['obs'], nc_data['latitude']),
             'slp_hpa':   (['obs'], nc_data['slp']),
             'category':  (['obs'], nc_data['category']),
             'storm_id':  (['obs'], nc_data['storm_id']),
             'timestep':  (['obs'], nc_data['timestep'])},
            coords={'obs': np.arange(len(nc_data['longitude']))},
            attrs={
                'model':           tdict['dataset']['model'],
                'experiment':      tdict['dataset']['exp'],
                'cat_method':      cat_method,
                'filter_category': str(category) if single_cat else 'all',
                'n_storms':        nstorms,
                'creation_date':   datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        ds['category'].attrs = {
            'long_name':     f'Saffir-Simpson category ({cat_method})',
            'flag_values':   '0 1 2 3 4 5',
            'flag_meanings': 'TD_TS Cat1 Cat2 Cat3 Cat4 Cat5'
        }
        ds['slp_hpa'].attrs = {'units': 'hPa', 'long_name': 'sea level pressure'}
        nc_path = os.path.join(tdict['paths']['plotdir'], f"{base}.nc")
        ds.to_netcdf(nc_path)
        print(f"✓ NetCDF → {nc_path}")
    except Exception as e:
        print(f"⚠ NetCDF save failed: {e}")
 
 
# ===============================================================================
# 3. DENSITY SCATTER (no category, unchanged)
# ===============================================================================
 
def plot_density_scatter(data_or_file, tdict, max_timesteps=None, sample_size=50000):
    """KDE-based density scatter — no category argument needed."""
    trajectories, _ = _get_data_from_input(data_or_file)
 
    lon_all, lat_all = [], []
    for sd in trajectories:
        lon_all.extend(sd['lon'])
        lat_all.extend(sd['lat'])
 
    lon_all = np.where(np.array(lon_all) > 180, np.array(lon_all) - 360, np.array(lon_all))
    lat_all = np.array(lat_all)
    total   = len(lon_all)
    print(f"Total TC points: {total:,}")
 
    if total > sample_size:
        idx      = np.random.choice(total, sample_size, replace=False)
        lon_s    = lon_all[idx]
        lat_s    = lat_all[idx]
    else:
        lon_s, lat_s = lon_all, lat_all
 
    try:
        kde      = gaussian_kde(np.vstack([lon_s, lat_s]))
        density  = kde(np.vstack([lon_s, lat_s]))
        dens_n   = density / density.max()
        order    = dens_n.argsort()
        lon_s    = lon_s[order]; lat_s = lat_s[order]; dens_n = dens_n[order]
    except Exception as e:
        print(f"KDE failed: {e}")
        dens_n = np.ones(len(lon_s))
 
    fig = plt.figure(figsize=(14, 8))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    sc  = ax.scatter(lon_s, lat_s, c=dens_n, s=8, alpha=0.7,
                     cmap='YlOrRd', norm=Normalize(0, 1),
                     transform=ccrs.PlateCarree(),
                     edgecolors='none', rasterized=True)
    ax.add_feature(cfeature.LAND,      color='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                        shrink=0.6, aspect=30, pad=0.08)
    cbar.set_label('Local Track Density (normalized)', fontsize=11, fontweight='bold')
 
    sd = tdict['time']['startdate']; ed = tdict['time']['enddate']
    m  = tdict['dataset']['model'];  ex = tdict['dataset']['exp']
    plt.title(f'TC Track Density Scatter\n{sd}–{ed} | {m} {ex} | n={total:,}',
              fontsize=13, fontweight='bold', pad=15)
 
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    base     = (f"density_scatter_{m.replace(' ','_')}_{ex.replace(' ','_')}"
                f"_{sd.replace('-','')}_{ed.replace('-','')}")
    pdf_path = os.path.join(tdict['paths']['plotdir'], f"{base}.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()
 
 
# ===============================================================================
# 4. GRIDDED TRACK DENSITY
# ===============================================================================
 
def plot_track_density_grid(data_or_file, tdict, grid_size=1,
                             max_timesteps=None, category=None,
                             cat_method='slp'):
    """
    Gridded TC track density (transits per month).
 
    Args:
        category   : int 0-5 → include only storms whose peak ≥ category; None = all
        cat_method : 'slp' or 'sshs'
    """
    trajectories, _ = _get_data_from_input(data_or_file)
    names  = _cat_names(cat_method)
    suffix = _method_suffix(cat_method)
 
    # ── category filter ───────────────────────────────────────────────────────
    if category is not None:
        trajectories = [s for s in trajectories
                        if _peak_category(s, cat_method) >= category]
        n_kept = len(trajectories)
        if n_kept == 0:
            print(f"⚠ No storms for {names[category]}+"); return
        print(f"✓ {n_kept} storms with peak ≥ {names[category]} ({cat_method})")
 
    lon_all, lat_all = [], []
    for sd in trajectories:
        lon_all.extend(sd['lon'])
        lat_all.extend(sd['lat'])
 
    lon_all = np.where(np.array(lon_all) > 180, np.array(lon_all) - 360, np.array(lon_all))
    lat_all = np.array(lat_all)
    total   = len(lon_all)
    print(f"Total TC points: {total:,}")
 
    if total == 0:
        print("No observations."); return
 
    # ── time period ───────────────────────────────────────────────────────────
    startdate = tdict['time']['startdate']
    enddate   = tdict['time']['enddate']
    fmt       = '%Y-%m-%d' if '-' in startdate else '%Y%m%d'
    start     = datetime.strptime(startdate, fmt)
    end       = datetime.strptime(enddate,   fmt)
    n_months  = (end.year - start.year) * 12 + (end.month - start.month) + 1
    print(f"Time period: {n_months} months")
 
    # ── grid ──────────────────────────────────────────────────────────────────
    lon_bins = np.arange(-180, 180 + grid_size, grid_size)
    lat_bins = np.arange(-50,   50 + grid_size, grid_size)
    counts, lon_edges, lat_edges = np.histogram2d(lon_all, lat_all,
                                                   bins=[lon_bins, lat_bins])
    tpm = counts / n_months   # transits per month
 
    vmax_d = tpm.max()
    vmin_d = tpm[tpm > 0].min() if np.any(tpm > 0) else 0.01
 
    if   vmax_d < 0.5: vmax_p = np.ceil(vmax_d * 20) / 20
    elif vmax_d < 1.0: vmax_p = np.ceil(vmax_d * 10) / 10
    elif vmax_d < 3.0: vmax_p = np.ceil(vmax_d * 2)  / 2
    else:              vmax_p = np.ceil(vmax_d)
    vmin_p = max(0.01, vmin_d)
 
    n_levels   = 12
    boundaries = np.logspace(np.log10(vmin_p), np.log10(vmax_p), n_levels + 1)
    boundaries[0] = 0.0
 
    disc = np.digitize(tpm, boundaries) - 1
    disc = np.clip(disc, 0, n_levels - 1)
    disc_masked = np.where(tpm > 0, disc, np.nan)
 
    base_colors = ['#FFFFFF', '#8B4513', '#D2691E', '#FFD700',
                   '#ADFF2F', '#00FF00', '#00CED1', '#0000FF']
    cmap_c  = LinearSegmentedColormap.from_list('tc_density', base_colors, N=256)
    cmap_d  = ListedColormap([cmap_c(i / n_levels) for i in range(n_levels)])
 
    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
 
    mesh = ax.pcolormesh(lon_edges, lat_edges, disc_masked.T,
                         cmap=cmap_d, vmin=0, vmax=n_levels,
                         transform=ccrs.PlateCarree(), shading='auto')
    ax.add_feature(cfeature.LAND,      color='lightgray', zorder=2, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--', zorder=4)
    gl.top_labels = False; gl.right_labels = False
 
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                        shrink=0.6, aspect=30, pad=0.08, extend='max')
    cbar.set_label('TC track density (transits per month)', fontsize=11, fontweight='bold')
    tick_pos = np.arange(n_levels) + 0.5
    cbar.set_ticks(tick_pos)
    tick_labels = []
    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == n_levels - 1:
            tick_labels.append(f'>{lo:.2g}')
        elif hi < 0.1:
            tick_labels.append(f'{lo:.3f}–{hi:.3f}')
        elif hi < 1.0:
            tick_labels.append(f'{lo:.2f}–{hi:.2f}')
        else:
            tick_labels.append(f'{lo:.2g}–{hi:.2g}')
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8, rotation=45)
 
    method_label  = 'wind-based (IBTrACS)' if cat_method == 'sshs' else 'SLP-based'
    category_str  = f' – {names[category]}+  [{method_label}]' if category is not None else ''
    m = tdict['dataset']['model']; ex = tdict['dataset']['exp']
    plt.title(f'TC Track Density (transits/month){category_str}\n'
              f'{startdate}–{enddate} | {m} {ex} | Grid: {grid_size}°',
              fontsize=13, fontweight='bold', pad=15)
 
    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    sd_c = startdate.replace('-', ''); ed_c = enddate.replace('-', '')
    m_c  = m.replace(' ', '_');        ex_c = ex.replace(' ', '_')
    cat_sfx = f'_cat{category}plus' if category is not None else ''
    base = f'track_density_grid{suffix}{cat_sfx}_{m_c}_{ex_c}_{sd_c}_{ed_c}'
 
    pdf_path = os.path.join(tdict['paths']['plotdir'], f'{base}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()
 
    # NetCDF
    try:
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        ds = xr.Dataset(
            {'transits_per_month': (['latitude', 'longitude'], tpm.T),
             'count':              (['latitude', 'longitude'], counts.T)},
            coords={'longitude': lon_centers, 'latitude': lat_centers},
            attrs={'model': m, 'experiment': ex,
                   'cat_method': cat_method,
                   'category_filter': names[category] if category is not None else 'all',
                   'grid_size_degrees': grid_size,
                   'n_months': n_months,
                   'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        nc_path = os.path.join(tdict['paths']['plotdir'], f'{base}.nc')
        ds.to_netcdf(nc_path)
        print(f"✓ NetCDF → {nc_path}")
    except Exception as e:
        print(f"⚠ NetCDF save failed: {e}")
 
 
# ===============================================================================
# 5. 6-PANEL DENSITY SCATTER BY CATEGORY
# ===============================================================================
 
def plot_density_scatter_by_category(data_or_file, tdict,
                                      max_timesteps=None, sample_size=10000,
                                      cat_method='slp'):
    """
    6-panel KDE density scatter, one panel per Saffir-Simpson category.
 
    Args:
        cat_method : 'slp' or 'sshs'
    """
    trajectories, _ = _get_data_from_input(data_or_file)
    names  = _cat_names(cat_method)
    suffix = _method_suffix(cat_method)
 
    points_by_cat = {c: {'lon': [], 'lat': []} for c in range(6)}
 
    for sd in trajectories:
        cats_ts = _get_cat_array(sd, cat_method)
        for i in range(len(sd['lon'])):
            lon = sd['lon'][i]
            lat = sd['lat'][i]
            if lon > 180: lon -= 360
            cat = int(cats_ts[i]) if i < len(cats_ts) else 0
            cat = np.clip(cat, 0, 5)
            points_by_cat[cat]['lon'].append(lon)
            points_by_cat[cat]['lat'].append(lat)
 
    for c in range(6):
        points_by_cat[c]['lon'] = np.array(points_by_cat[c]['lon'])
        points_by_cat[c]['lat'] = np.array(points_by_cat[c]['lat'])
 
    print(f"\nPoints per category ({cat_method}):")
    for c in range(6):
        print(f"  Cat {c}: {len(points_by_cat[c]['lon']):,}")
 
    cmaps = ['Greens', 'YlGn', 'YlOrBr', 'Oranges', 'OrRd', 'RdPu']
    method_label = 'wind-based (IBTrACS)' if cat_method == 'sshs' else 'SLP-based'
 
    fig = plt.figure(figsize=(18, 12))
 
    for c in range(6):
        ax = plt.subplot(2, 3, c + 1, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
 
        lon_c = points_by_cat[c]['lon']
        lat_c = points_by_cat[c]['lat']
        n_pts = len(lon_c)
 
        ax.add_feature(cfeature.LAND,      color='lightgray', zorder=0, alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
 
        if n_pts == 0:
            ax.set_title(f'{names[c]}\n(no obs)', fontsize=10, fontweight='bold')
            continue
 
        lon_s = lon_c[np.random.choice(n_pts, min(n_pts, sample_size), replace=False)]
        lat_s = lat_c[np.random.choice(n_pts, min(n_pts, sample_size), replace=False)]
 
        try:
            kde    = gaussian_kde(np.vstack([lon_s, lat_s]))
            dens   = kde(np.vstack([lon_s, lat_s]))
            dens_n = dens / dens.max()
            sc = ax.scatter(lon_s, lat_s, c=dens_n, s=5, alpha=0.7,
                            cmap=cmaps[c], norm=Normalize(0, 1),
                            transform=ccrs.PlateCarree(),
                            edgecolors='none', rasterized=True)
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                                shrink=0.9, aspect=20, pad=0.05)
            cbar.set_label('Norm. Density', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        except Exception as e:
            print(f"  Cat {c}: KDE failed ({e})")
            ax.scatter(lon_s, lat_s, s=3, alpha=0.5, color='steelblue',
                       transform=ccrs.PlateCarree())
 
        gl = ax.gridlines(draw_labels=(c >= 3), linewidth=0.3,
                          color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        ax.set_title(f'{names[c]}\n(n={n_pts:,})', fontsize=10, fontweight='bold')
 
    sd = tdict['time']['startdate']; ed = tdict['time']['enddate']
    m  = tdict['dataset']['model'];  ex = tdict['dataset']['exp']
    plt.suptitle(f'TC Track Density by Category  [{method_label}]\n{sd}–{ed} | {m} {ex}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
 
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    base = (f"density_scatter_by_cat{suffix}_{m.replace(' ','_')}"
            f"_{ex.replace(' ','_')}_{sd.replace('-','')}_{ed.replace('-','')}")
    pdf_path = os.path.join(tdict['paths']['plotdir'], f"{base}.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()
 
 
# ===============================================================================
# 6. DURATION DISTRIBUTION (unchanged — no category)
# ===============================================================================
 
def plot_tc_duration_distribution(data_or_file, tdict):
    """TC lifetime histogram + KDE with mean/median lines."""
    trajectories, _ = _get_data_from_input(data_or_file)
 
    durations = []
    for sd in trajectories:
        try:
            t0 = datetime(sd['year'][0],  sd['month'][0],  sd['day'][0],  sd['hour'][0])
            t1 = datetime(sd['year'][-1], sd['month'][-1], sd['day'][-1], sd['hour'][-1])
            durations.append((t1 - t0).total_seconds() / 86400)
        except Exception:
            continue
 
    durations = np.array(durations)
    print(f"Storms: {len(durations)}  |  mean {durations.mean():.1f} d  "
          f"| median {np.median(durations):.1f} d  |  max {durations.max():.1f} d")
 
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(durations, bins=40, color='steelblue', alpha=0.7,
            edgecolor='black', density=True, label='Histogram')
    kde = gaussian_kde(durations)
    xr_ = np.linspace(durations.min(), durations.max(), 200)
    ax.plot(xr_, kde(xr_), 'r-', linewidth=2.5, label='KDE')
    ax.axvline(durations.mean(),        color='green',  linestyle='--', linewidth=2,
               label=f'Mean: {durations.mean():.1f} d')
    ax.axvline(np.median(durations),    color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(durations):.1f} d')
    ax.set_xlabel('Duration (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    sd = tdict['time']['startdate']; ed = tdict['time']['enddate']
    ax.set_title(f"TC Lifetime Distribution\n{sd}–{ed} | {tdict['dataset']['model']}",
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, linestyle=':')
    plt.tight_layout()
 
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    m  = tdict['dataset']['model'].replace(' ', '_')
    ex = tdict['dataset']['exp'].replace(' ', '_')
    pdf_path = os.path.join(tdict['paths']['plotdir'],
                            f'tc_duration_{m}_{ex}_{sd}_{ed}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()
 
 
# ===============================================================================
# 7. DURATION BY CATEGORY
# ===============================================================================
 
def plot_tc_duration_by_category(data_or_file, tdict, cat_method='slp'):
    """
    Normalized duration histograms per Saffir-Simpson category.
 
    Args:
        cat_method : 'slp' or 'sshs'
    """
    trajectories, _ = _get_data_from_input(data_or_file)
    names  = _cat_names(cat_method)
    suffix = _method_suffix(cat_method)
    method_label = 'wind-based (IBTrACS)' if cat_method == 'sshs' else 'SLP-based'
 
    dur_by_cat = {c: [] for c in range(6)}
 
    for sd in trajectories:
        try:
            t0 = datetime(sd['year'][0],  sd['month'][0],  sd['day'][0],  sd['hour'][0])
            t1 = datetime(sd['year'][-1], sd['month'][-1], sd['day'][-1], sd['hour'][-1])
            dur = (t1 - t0).total_seconds() / 86400
        except Exception:
            continue
        peak = _peak_category(sd, cat_method)
        dur_by_cat[peak].append(dur)
 
    colors    = ['#1b9e77', '#d95f02', '#e6ab02', '#e7298a', '#a6761d', '#7570b3']
    max_dur   = max(max(v) for v in dur_by_cat.values() if v)
    bins      = np.arange(0, int(np.ceil(max_dur)) + 2, 1)
 
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in range(6):
        data = dur_by_cat[c]
        if not data:
            continue
        ax.hist(data, bins=bins, histtype='step', linewidth=2.2,
                color=colors[c], density=True, alpha=0.85,
                label=f'{names[c]} (n={len(data)})')
 
    sd = tdict['time']['startdate']; ed = tdict['time']['enddate']
    ax.set_xlabel('TC duration (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability density', fontsize=12, fontweight='bold')
    ax.set_title(f'Normalized TC Duration by Category  [{method_label}]\n{sd}–{ed}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, frameon=False)
    ax.grid(alpha=0.25, linestyle=':')
    plt.tight_layout()
 
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    m  = tdict['dataset']['model'].replace(' ', '_')
    ex = tdict['dataset']['exp'].replace(' ', '_')
    base     = f'tc_duration_by_cat{suffix}_{m}_{ex}_{sd}_{ed}'
    pdf_path = os.path.join(tdict['paths']['plotdir'], f'{base}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ PDF  → {pdf_path}")
    plt.show(); plt.close()
 
 
# ===============================================================================
# 8. BASIN DOUGHNUT
# ===============================================================================
 
def plot_tc_basin_doughnut(data_or_file, tdict, reference_freq=90.0,
                            cat_method='slp'):
    """
    Doughnut chart of annual TC frequency by basin + comparison bar chart.
 
    Args:
        reference_freq : reference NH frequency for scaling (default 90)
        cat_method     : 'slp' or 'sshs' — affects title/filename only here,
                         since basin assignment uses genesis position regardless
    """
    trajectories, _ = _get_data_from_input(data_or_file)
    suffix       = _method_suffix(cat_method)
    method_label = 'wind-based (IBTrACS)' if cat_method == 'sshs' else 'SLP-based'
 
    nh_season = [5, 6, 7, 8, 9, 10, 11]
    sh_season = [10, 11, 12, 1, 2, 3, 4, 5]
    nh_basins = ['North Atlantic', 'East Pacific', 'West Pacific', 'North Indian']
    sh_basins = ['South Indian', 'South Pacific']
 
    basin_counts = {b: set() for b in nh_basins + sh_basins}
    years_seen   = set()
 
    for sd in trajectories:
        sid   = sd['id']
        year  = sd['year'][0]
        month = sd['month'][0]
        lon   = sd['lon'][0]
        lat   = sd['lat'][0]
        years_seen.add(year)
 
        basin = get_basin_ibtracs(lon, lat)
        hemi  = 'NH' if lat >= 0 else 'SH'
 
        if hemi == 'NH' and month in nh_season and basin in nh_basins:
            basin_counts[basin].add(sid)
        elif hemi == 'SH' and month in sh_season and basin in sh_basins:
            basin_counts[basin].add(sid)
 
    n_years    = len(years_seen)
    basin_freq = {b: len(s) / n_years for b, s in basin_counts.items()}
    nh_total   = sum(basin_freq[b] for b in nh_basins)
    sh_total   = sum(basin_freq[b] for b in sh_basins)
 
    print(f"Period: {min(years_seen)}–{max(years_seen)}  ({n_years} yr)")
    print(f"NH total: {nh_total:.1f}/yr   SH total: {sh_total:.1f}/yr")
    for b in nh_basins + sh_basins:
        print(f"  {b}: {basin_freq[b]:.1f}")
 
    basins_ord = ['North Atlantic', 'East Pacific', 'West Pacific',
                  'North Indian',   'South Indian', 'South Pacific']
    colors     = {'North Atlantic': '#3498db', 'East Pacific': '#2ecc71',
                  'West Pacific':   '#e74c3c', 'North Indian': '#f39c12',
                  'South Indian':   '#9b59b6', 'South Pacific': '#34495e'}
 
    sizes      = [basin_freq[b] for b in basins_ord]
    col_list   = [colors[b]     for b in basins_ord]
    scale      = nh_total / reference_freq
    r_in       = 0.20
    r_out      = 0.20 + 0.28 * scale
    total_freq = sum(sizes)
    pcts       = [100 * basin_freq[b] / total_freq for b in basins_ord]
 
    # ── doughnut ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, _ = ax.pie(sizes, colors=col_list, startangle=90,
                       counterclock=False, radius=r_out,
                       wedgeprops=dict(width=r_out - r_in,
                                       edgecolor='white', linewidth=2))
 
    for wedge, pct in zip(wedges, pcts):
        if pct < 3.0: continue
        angle  = np.deg2rad(0.5 * (wedge.theta1 + wedge.theta2))
        r_text = r_in + 0.5 * (r_out - r_in)
        ax.text(r_text * np.cos(angle), r_text * np.sin(angle),
                f'{pct:.0f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=12)
 
    from matplotlib.patches import Circle
    ax.add_artist(Circle((0, 0), r_in, color='white', zorder=10))
    ax.text(0,  0.08, f'{nh_total:.1f}', ha='center', va='center',
            fontsize=32, fontweight='bold', color='black',  zorder=11)
    ax.text(0, -0.08, f'{sh_total:.1f}', ha='center', va='center',
            fontsize=28, fontweight='bold', color='gray',   zorder=11)
    ax.text(0,  0.15, 'NH', ha='center', va='center',
            fontsize=20, fontweight='bold', color='black',  zorder=11)
    ax.text(0, -0.15, 'SH', ha='center', va='center',
            fontsize=20, fontweight='bold', color='gray',   zorder=11)
 
    sd = tdict['time']['startdate']; ed = tdict['time']['enddate']
    ax.set_title(f'TC Frequency by Basin  [{method_label}]\n'
                 f'(storms/yr, {min(years_seen)}–{max(years_seen)})\n'
                 f'NH: May–Nov | SH: Oct–May',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(wedges, [f"{b}: {basin_freq[b]:.1f}/yr" for b in basins_ord],
              loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.axis('equal')
    plt.tight_layout()
 
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    m_c  = tdict['dataset']['model'].replace(' ', '_')
    ex_c = tdict['dataset']['exp'].replace(' ', '_')
    sd_c = sd.replace('-', ''); ed_c = ed.replace('-', '')
    base_d   = f'tc_doughnut{suffix}_{m_c}_{ex_c}_{sd_c}_{ed_c}'
    pdf_d    = os.path.join(tdict['paths']['plotdir'], f'{base_d}.pdf')
    plt.savefig(pdf_d, dpi=300, bbox_inches='tight')
    print(f"✓ Doughnut → {pdf_d}")
    plt.show(); plt.close()
 
    # ── comparison bar chart ──────────────────────────────────────────────────
    roberts = {'North Atlantic': 12.5, 'East Pacific': 16.0, 'West Pacific': 25.0,
               'North Indian':    5.5, 'South Indian': 10.0, 'South Pacific':  4.5}
 
    print(f"\n{'Basin':<20} {'This study':>12} {'Roberts+20':>12} {'Diff':>8}")
    print('-' * 55)
    for b in basins_ord:
        diff = basin_freq[b] - roberts.get(b, np.nan)
        print(f"{b:<20} {basin_freq[b]:>12.1f} {roberts.get(b, np.nan):>12.1f} {diff:>8.1f}")
 
    fig, ax = plt.subplots(figsize=(12, 6))
    x  = np.arange(len(basins_ord)); w = 0.35
    ax.bar(x - w/2, [basin_freq[b] for b in basins_ord], w,
           label='This study',           color='steelblue', edgecolor='black')
    ax.bar(x + w/2, [roberts.get(b, 0)  for b in basins_ord], w,
           label='Roberts et al. (2020)', color='coral',     edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace(' ', '\n') for b in basins_ord], fontsize=10)
    ax.set_xlabel('Ocean Basin',     fontsize=12, fontweight='bold')
    ax.set_ylabel('Storms per year', fontsize=12, fontweight='bold')
    ax.set_title(f'TC Frequency: This Study vs Roberts et al. (2020)  [{method_label}]',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
 
    base_c   = f'tc_freq_comparison{suffix}_{m_c}_{ex_c}_{sd_c}_{ed_c}'
    pdf_c    = os.path.join(tdict['paths']['plotdir'], f'{base_c}.pdf')
    plt.savefig(pdf_c, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison → {pdf_c}")
    plt.show(); plt.close()