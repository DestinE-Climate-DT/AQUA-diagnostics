# plotting_TCs_custom_memory.py
# 
# TROPICAL CYCLONE VISUALIZATION - MEMORY-OPTIMIZED VERSION
# ==========================================================
#
# This module provides plotting functions for TC trajectories that accept
# BOTH file paths (backward compatible) AND TCDataManager objects (memory-
# optimized, 100-1000x faster for multiple plots).
#
# MAIN FUNCTIONS:
# ---------------
# 1. plot_trajectories_direct(data_or_file, tdict, max_timesteps)
#    Simple scatter plot of all TC tracks (black dots). Quick diagnostic view.
#
# 2. plot_trajectories_colored(data_or_file, tdict, color_by, max_timesteps)
#    TC tracks as continuous lines colored by intensity (SLP) or Saffir-Simpson
#    category. Handles dateline crossings.
#
# 3. plot_trajectories_by_category(data_or_file, tdict, category, max_timesteps)
#    Plot only TCs of a specific category (0=TD, 1-5=Cat 1-5) based on peak
#    intensity during lifetime.
#
# 4. plot_density_scatter(data_or_file, tdict, max_timesteps, sample_size)
#    KDE-based density scatter where each point is colored by local density
#    (normalized 0-1). Points sorted by density for better visualization.
#
# 5. plot_track_density_grid(data_or_file, tdict, grid_size, max_timesteps)
#    Gridded density map showing "transits per month" following HighResMIP-
#    PRIMAVERA standards. Discrete logarithmic colorbar, custom colormap.
#
# 6. plot_density_scatter_by_category(data_or_file, tdict, max_timesteps, sample_size)
#    6-panel subplot with KDE density for each Saffir-Simpson category
#    (TD through Cat 5). Horizontal colorbars.
#
# USAGE:
# ------
# Option 1 (traditional, slow):
#   plot_trajectories_direct("filtered_file.txt", config)
#
# Option 2 (memory-optimized, fast):
#   from tc_data_manager import TCDataManager
#   tc_data = TCDataManager("filtered_file.txt")  # Load once
#   plot_trajectories_direct(tc_data, config)      # Instant!
#   plot_density_scatter(tc_data, config)          # Instant!
#   plot_track_density_grid(tc_data, config)       # Instant!
#
# All functions produce publication-ready PDF outputs with Cartopy PlateCarree
# projection and standardized naming conventions.
# ===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import os
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap, LogNorm
from scipy.stats import gaussian_kde
from datetime import datetime


# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def _get_data_from_input(data_or_file):
    """
    Helper to handle both file paths and TCDataManager objects.
    
    Args:
        data_or_file: Either a file path (string) or TCDataManager object
    
    Returns:
        trajectories_dict: Dictionary of trajectories by storm_id
        data_dict: Full data dictionary (or None if from file)
    """
    # Check if it's a TCDataManager object
    if hasattr(data_or_file, 'get_all_points'):
        print("Using data from memory (fast)")
        trajectories = data_or_file.get_trajectories()
        data_dict = data_or_file.get_all_data()
        return trajectories, data_dict
    else:
        print("Reading from file (slow)...")
        _, _, trajectories = getTrajectories_direct(data_or_file)
        return trajectories, None


def category_from_slp_pa(slp_pa):
    """Saffir-Simpson Category from SLP Pascal"""
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


def get_basin_ibtracs(lon, lat):
    """Classification based on IBTrACS (WMO standard)"""
    lon_360 = lon if lon >= 0 else lon + 360
    
    if (260 <= lon_360 <= 360 or 0 <= lon_360 <= 0) and 0 <= lat <= 70:
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
    Read trajectories directly from file (fallback for file input).
    
    Returns:
        numtraj, maxNumPts, trajectories
    """
    print(f"Reading trajectories from file: {filename}")
    
    trajectories = {}
    
    with open(filename, 'r') as f:
        for line in f:
            if 'track_id' in line or 'year' in line:
                continue
            
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            
            storm_id = parts[0]
            year = int(parts[1])
            month = int(parts[2])
            lon = float(parts[7])
            lat = float(parts[8])
            slp = float(parts[9])
            wind = float(parts[10])
            
            if storm_id not in trajectories:
                trajectories[storm_id] = {
                    'lon': [],
                    'lat': [],
                    'slp': [],
                    'wind': [],
                    'year': [],
                    'month': []
                }
            
            trajectories[storm_id]['lon'].append(lon)
            trajectories[storm_id]['lat'].append(lat)
            trajectories[storm_id]['slp'].append(slp)
            trajectories[storm_id]['wind'].append(wind)
            trajectories[storm_id]['year'].append(year)
            trajectories[storm_id]['month'].append(month)
    
    # Convert to numpy
    for storm_id in trajectories:
        for key in trajectories[storm_id]:
            trajectories[storm_id][key] = np.array(trajectories[storm_id][key])
    
    numtraj = len(trajectories)
    maxNumPts = max([len(t['lon']) for t in trajectories.values()])
    
    print(f"Found {numtraj} trajectories")
    
    return numtraj, maxNumPts, trajectories


# ===============================================================================
# PLOTTING FUNCTIONS
# ===============================================================================

def plot_trajectories_direct(data_or_file, tdict, max_timesteps=None):
    """
    Plot all TC trajectories as simple scatter points.
    
    Args:
        data_or_file: TCDataManager object OR file path string
        tdict: Configuration dictionary
        max_timesteps: Optional timestep limit
    """
    # Get data (from memory OR file)
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    nstorms = len(trajectories)
    
    # Initialize figure
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    # Title
    plt.title(
        f"TC Tracks - {tdict['dataset']['model']} - {tdict['dataset']['exp']}\n"
        f"({nstorms} tracks, {tdict['time']['startdate']} – {tdict['time']['enddate']})",
        fontsize=14, fontweight='bold'
    )
    
    # Geographic features
    ax.add_feature(cfeature.LAND, color='lightgrey', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 12, 'color': 'black'}
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    
    # Plot each trajectory
    for storm_id, data in trajectories.items():
        lon = data['lon']
        lat = data['lat']
        
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
        
        lon_plot = np.where(lon > 180, lon - 360, lon)
        
        ax.scatter(
            x=lon_plot,
            y=lat,
            color="black",
            s=22,
            linewidths=0.5,
            marker=".",
            alpha=0.9,
            transform=ccrs.PlateCarree()
        )
    
    # Save figure
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f"tracks_{tdict['dataset']['model']}_{tdict['dataset']['exp']}_{startdate}_{enddate}.pdf"
    )
    
    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    print(f"✓ Plot saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_trajectories_colored(data_or_file, tdict, color_by='intensity', max_timesteps=None):
    """
    Plot trajectories colored by intensity or category.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        color_by: 'intensity' or 'category'
        max_timesteps: Optional timestep limit
    """
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    nstorms = len(trajectories)
    
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, color='lightgrey', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    
    # Colormap settings
    if color_by == 'intensity':
        cmap = plt.cm.YlOrRd
        vmin, vmax = 920, 1010
        label = 'SLP (hPa)'
    else:
        cmap = plt.cm.RdYlGn_r
        vmin, vmax = 0, 5
        label = 'Saffir–Simpson Category'
    
    # Plot trajectories
    for storm_id, data in trajectories.items():
        lon = data['lon']
        lat = data['lat']
        slp = data['slp'] / 100.0
        
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
            slp = slp[:max_timesteps]
        
        lon_plot = np.where(lon > 180, lon - 360, lon)
        
        if len(lon_plot) < 2:
            continue
        
        dlon = np.abs(np.diff(lon_plot))
        valid = dlon < 180
        
        min_slp = np.nanmin(slp)
        cat = category_from_slp_pa(min_slp * 100)
        values = np.full(len(lon_plot) - 1, cat)
        
        points = np.array([lon_plot, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        segments = segments[valid]
        values = values[valid]
        
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            linewidth=1.3,
            alpha=0.9,
            transform=ccrs.PlateCarree()
        )
        lc.set_array(values)
        ax.add_collection(lc)
    
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=12)
    
    if color_by == 'category':
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['TD', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'])
    
    plt.title(
        f"TC Tracks coloured by {color_by}\n"
        f"{tdict['dataset']['model']} - {tdict['dataset']['exp']} ({nstorms} tracks)",
        fontsize=14,
        fontweight='bold'
    )
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    model_clean = tdict['dataset']['model'].replace(" ", "_")
    exp_clean = tdict['dataset']['exp'].replace(" ", "_")
    color_clean = str(color_by).replace(" ", "_")
    
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f"tracks_colored_{color_clean}_{model_clean}_{exp_clean}_{startdate}_{enddate}.pdf"
    )
    
    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    print(f"✓ Plot saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_trajectories_by_category(data_or_file, tdict, category=1, max_timesteps=None):
    """
    Plot only TCs of a specific category.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        category: 0-5 (TD to Cat 5)
        max_timesteps: Optional timestep limit
    """
    cat_names = [
        'TD (≥1005 hPa)', 'Cat 1 (990–1004)', 'Cat 2 (975–989)',
        'Cat 3 (960–974)', 'Cat 4 (945–959)', 'Cat 5 (<945)'
    ]
    cat_colors = ['lightgreen', 'gold', 'orange', 'red', 'darkred', 'purple']
    
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    # Filter by category
    filtered_trajectories = {}
    for storm_id, data in trajectories.items():
        min_slp = np.min(data['slp'])
        peak_category = category_from_slp_pa(min_slp)
        
        if peak_category == category:
            filtered_trajectories[storm_id] = data
    
    n_filtered = len(filtered_trajectories)
    
    if n_filtered == 0:
        print(f"⚠ No trajectories found for {cat_names[category]}")
        return
    
    print(f"✓ Found {n_filtered} trajectories for {cat_names[category]}")
    
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, color='lightgrey', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    for storm_id, data in filtered_trajectories.items():
        lon = data['lon']
        lat = data['lat']
        
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
        
        lon_plot = np.where(lon > 180, lon - 360, lon)
        lat_plot = lat.copy()
        
        dlon = np.diff(lon_plot)
        split_indices = np.where(np.abs(dlon) > 180)[0]
        
        lon_plot_fixed = lon_plot.copy()
        lat_plot_fixed = lat_plot.copy()
        for idx in split_indices:
            lon_plot_fixed[idx+1] = np.nan
            lat_plot_fixed[idx+1] = np.nan
        
        ax.plot(
            lon_plot_fixed,
            lat_plot_fixed,
            color=cat_colors[category],
            linewidth=1.5,
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )
    
    plt.title(
        f"TC Tracks – {cat_names[category]}\n"
        f"{tdict['dataset']['model']} - {tdict['dataset']['exp']} ({n_filtered} cyclones)",
        fontsize=14,
        fontweight='bold'
    )
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    
    model_clean = tdict['dataset']['model'].replace(" ", "_")
    exp_clean = tdict['dataset']['exp'].replace(" ", "_")
    startdate_clean = tdict['time']['startdate'].replace(" ", "").replace("-", "")
    enddate_clean = tdict['time']['enddate'].replace(" ", "").replace("-", "")
    
    save_filename = (
        f"tracks_cat{category}_{model_clean}_{exp_clean}_"
        f"{startdate_clean}_{enddate_clean}.pdf"
    )
    
    save_path = os.path.join(tdict['paths']['plotdir'], save_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    print(f"✓ Plot saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_density_scatter(data_or_file, tdict, max_timesteps=None, sample_size=50000):
    """
    Plot TC track density scatter with KDE coloring and point sorting.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        max_timesteps: Optional timestep limit
        sample_size: Max points for KDE (default 50000)
    """
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    # Extract all points
    lon_all = []
    lat_all = []
    for storm_data in trajectories.values():
        lon_all.extend(storm_data['lon'])
        lat_all.extend(storm_data['lat'])
    
    lon_all = np.array(lon_all)
    lat_all = np.array(lat_all)
    
    # Convert longitude
    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    
    total_points = len(lon_all)
    print(f"Total TC observation points: {total_points:,}")
    
    # Sample if needed
    if total_points > sample_size:
        print(f"Sampling {sample_size:,} points for KDE...")
        indices = np.random.choice(total_points, sample_size, replace=False)
        lon_sample = lon_all[indices]
        lat_sample = lat_all[indices]
    else:
        lon_sample = lon_all
        lat_sample = lat_all
    
    # Compute KDE
    print("Computing KDE...")
    try:
        xy = np.vstack([lon_sample, lat_sample])
        kde = gaussian_kde(xy)
        density = kde(xy)
        density_normalized = density / density.max()
        
        # Sort by density
        idx = density_normalized.argsort()
        lon_sorted = lon_sample[idx]
        lat_sorted = lat_sample[idx]
        density_sorted = density_normalized[idx]
    except Exception as e:
        print(f"KDE failed: {e}")
        lon_sorted = lon_sample
        lat_sorted = lat_sample
        density_sorted = np.ones(len(lon_sample))
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    scatter = ax.scatter(
        lon_sorted, lat_sorted,
        c=density_sorted,
        s=8,
        alpha=0.7,
        cmap='YlOrRd',
        norm=Normalize(vmin=0, vmax=1),
        transform=ccrs.PlateCarree(),
        edgecolors='none',
        rasterized=True
    )
    
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    
    cbar = plt.colorbar(
        scatter,
        ax=ax,
        orientation='horizontal',
        shrink=0.6,
        aspect=30,
        pad=0.08
    )
    cbar.set_label('Local Track Density (normalized)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    model = tdict['dataset']['model']
    exp = tdict['dataset']['exp']
    
    plt.title(
        f'TC Track Density Scatter Plot\n'
        f'{startdate}–{enddate} | {model} {exp} | n={total_points:,} obs',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    
    model_clean = model.replace(" ", "_")
    exp_clean = exp.replace(" ", "_")
    startdate_clean = startdate.replace(" ", "").replace("-", "")
    enddate_clean = enddate.replace(" ", "").replace("-", "")
    
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'density_scatter_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}.pdf'
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_track_density_grid(data_or_file, tdict, grid_size=2.5, max_timesteps=None):
    """
    Plot TC track density as transits per month with discrete colorbar.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        grid_size: Grid cell size in degrees (default 2.5)
        max_timesteps: Optional timestep limit
    """
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    # Extract all points
    lon_all = []
    lat_all = []
    for storm_data in trajectories.values():
        lon_all.extend(storm_data['lon'])
        lat_all.extend(storm_data['lat'])
    
    lon_all = np.array(lon_all)
    lat_all = np.array(lat_all)
    
    # Convert longitude
    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    
    total_points = len(lon_all)
    print(f"Total TC observation points: {total_points:,}")
    
    # Calculate time period
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    if '-' in startdate:
        start = datetime.strptime(startdate, '%Y-%m-%d')
        end = datetime.strptime(enddate, '%Y-%m-%d')
    else:
        start = datetime.strptime(startdate, '%Y%m%d')
        end = datetime.strptime(enddate, '%Y%m%d')
    
    n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    print(f"Time period: {n_months} months")
    
    # Create grid
    lon_bins = np.arange(-180, 180 + grid_size, grid_size)
    lat_bins = np.arange(-50, 50 + grid_size, grid_size)
    
    counts, lon_edges, lat_edges = np.histogram2d(
        lon_all, lat_all,
        bins=[lon_bins, lat_bins]
    )
    
    transits_per_month = counts / n_months
    
    print(f"Max transits/month: {transits_per_month.max():.2f}")
    
    # Automatic boundaries
    vmax = transits_per_month.max()
    vmin = transits_per_month[transits_per_month > 0].min() if np.any(transits_per_month > 0) else 0.01
    
    n_levels = 12
    boundaries = np.logspace(np.log10(max(vmin, 0.01)), np.log10(max(vmax, 1.0)), n_levels + 1)
    boundaries[0] = 0.0
    
    # Discretize
    transits_discrete = np.digitize(transits_per_month, boundaries) - 1
    transits_discrete = np.clip(transits_discrete, 0, n_levels - 1)
    transits_discrete_masked = np.where(transits_per_month > 0, transits_discrete, np.nan)
    
    # Colormap
    base_colors = ['#FFFFFF', '#8B4513', '#D2691E', '#FFD700', '#ADFF2F', '#00FF00', '#00CED1', '#0000FF']
    cmap_continuous = LinearSegmentedColormap.from_list('tc_density', base_colors, N=256)
    colors_discrete = [cmap_continuous(i / n_levels) for i in range(n_levels)]
    cmap_discrete = ListedColormap(colors_discrete)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, transits_discrete_masked.T,
        cmap=cmap_discrete,
        vmin=0,
        vmax=n_levels,
        transform=ccrs.PlateCarree(),
        shading='auto'
    )
    
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=2, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--',
        zorder=4
    )
    gl.top_labels = False
    gl.right_labels = False
    
    cbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation='horizontal',
        shrink=0.6,
        aspect=30,
        pad=0.08,
        extend='max'
    )
    cbar.set_label('TC track density (transits per month)', fontsize=11, fontweight='bold')
    
    tick_positions = np.arange(n_levels) + 0.5
    cbar.set_ticks(tick_positions)
    
    tick_labels = []
    for i in range(n_levels):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == n_levels - 1:
            tick_labels.append(f'>{lower:.2g}')
        else:
            tick_labels.append(f'{lower:.2g}–{upper:.2g}')
    
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8, rotation=45)
    
    model = tdict['dataset']['model']
    exp = tdict['dataset']['exp']
    
    plt.title(
        f'TC Track Density (transits per month)\n'
        f'{startdate}–{enddate} | {model} {exp} | Grid: {grid_size}°',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    
    # Clean names (remove spaces)
    model_clean = model.replace(" ", "_")
    exp_clean = exp.replace(" ", "_")
    startdate_clean = startdate.replace(" ", "").replace("-", "")
    enddate_clean = enddate.replace(" ", "").replace("-", "")
    
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'track_density_grid_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}.pdf'
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Track density grid saved to: {save_path}")
    
    plt.show()
    plt.close()
     


def plot_density_scatter_by_category(data_or_file, tdict, max_timesteps=None, sample_size=10000):
    """
    Plot 6-panel density scatter by Saffir-Simpson category.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        max_timesteps: Optional timestep limit
        sample_size: Max points per category for KDE (default 10000)
    """
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    # Classify points by category
    points_by_category = {cat: {'lon': [], 'lat': []} for cat in range(6)}
    
    for storm_data in trajectories.values():
        for i in range(len(storm_data['lon'])):
            lon = storm_data['lon'][i]
            lat = storm_data['lat'][i]
            slp = storm_data['slp'][i]
            
            if lon > 180:
                lon -= 360
            
            cat = category_from_slp_pa(slp)
            
            points_by_category[cat]['lon'].append(lon)
            points_by_category[cat]['lat'].append(lat)
    
    # Convert to numpy
    for cat in range(6):
        points_by_category[cat]['lon'] = np.array(points_by_category[cat]['lon'])
        points_by_category[cat]['lat'] = np.array(points_by_category[cat]['lat'])
    
    print("\nPoints per category:")
    for cat in range(6):
        n_pts = len(points_by_category[cat]['lon'])
        print(f"  Cat {cat}: {n_pts:,} points")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    cat_names = [
        'TD (≥1005 hPa)', 'Cat 1 (990–1004 hPa)', 'Cat 2 (975–989 hPa)',
        'Cat 3 (960–974 hPa)', 'Cat 4 (945–959 hPa)', 'Cat 5 (<945 hPa)'
    ]
    
    cmaps = ['Greens', 'YlGn', 'YlOrBr', 'Oranges', 'OrRd', 'RdPu']
    
    for cat in range(6):
        ax = plt.subplot(2, 3, cat + 1, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
        
        lon_cat = points_by_category[cat]['lon']
        lat_cat = points_by_category[cat]['lat']
        n_points = len(lon_cat)
        
        if n_points == 0:
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.set_title(f'{cat_names[cat]}\n(no observations)', fontsize=11, fontweight='bold')
            continue
        
        # Sample if needed
        if n_points > sample_size:
            indices = np.random.choice(n_points, sample_size, replace=False)
            lon_sample = lon_cat[indices]
            lat_sample = lat_cat[indices]
        else:
            lon_sample = lon_cat
            lat_sample = lat_cat
        
        # Compute KDE
        try:
            xy = np.vstack([lon_sample, lat_sample])
            kde = gaussian_kde(xy)
            density = kde(xy)
            density_normalized = density / density.max()
            
            scatter = ax.scatter(
                lon_sample, lat_sample,
                c=density_normalized,
                s=5,
                alpha=0.7,
                cmap=cmaps[cat],
                norm=Normalize(vmin=0, vmax=1),
                transform=ccrs.PlateCarree(),
                edgecolors='none',
                rasterized=True
            )
            
            cbar = plt.colorbar(
                scatter,
                ax=ax,
                orientation='horizontal',
                shrink=0.9,
                aspect=20,
                pad=0.05
            )
            cbar.set_label('Normalized Density', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
        except Exception as e:
            print(f"  Cat {cat}: KDE failed ({e})")
            ax.scatter(
                lon_sample, lat_sample,
                s=3, alpha=0.5, color='blue',
                transform=ccrs.PlateCarree()
            )
        
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0, alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        
        gl = ax.gridlines(
            draw_labels=True if cat >= 3 else False,
            linewidth=0.3,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(f'{cat_names[cat]}\n(n={n_points:,} obs)', fontsize=11, fontweight='bold')
    
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    model = tdict['dataset']['model']
    exp = tdict['dataset']['exp']
    
    plt.suptitle(
        f'TC Track Density by Saffir-Simpson Category\n'
        f'{startdate}–{enddate} | {model} {exp}',
        fontsize=15,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    
    model_clean = model.replace(" ", "_")
    exp_clean = exp.replace(" ", "_")
    startdate_clean = startdate.replace(" ", "").replace("-", "")
    enddate_clean = enddate.replace(" ", "").replace("-", "")
    
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'density_scatter_by_category_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}.pdf'
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 6-panel plot saved: {save_path}")
    
    plt.show()
    plt.close()