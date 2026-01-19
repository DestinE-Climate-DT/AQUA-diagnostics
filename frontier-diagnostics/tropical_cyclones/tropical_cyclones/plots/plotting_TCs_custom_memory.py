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
#    OR plot a specified category: 
#    plot_trajectories_colored(data_or_file, tdict, color_by,category=None, max_timesteps=None)
#    Categories 0=TD, 1-5=Cat 1-5 based on peak intensity during lifetime.
#
# 3. plot_density_scatter(data_or_file, tdict, max_timesteps, sample_size)
#    KDE-based density scatter where each point is colored by local density
#    (normalized 0-1). Points sorted by density for better visualization.
#
# 4. plot_track_density_grid(data_or_file, tdict, grid_size, max_timesteps)
#    Gridded density map showing "transits per month" following HighResMIP-
#    PRIMAVERA standards. Discrete logarithmic colorbar, custom colormap.
#
# 5. plot_density_scatter_by_category(data_or_file, tdict, max_timesteps, sample_size)
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
# projection and standardized naming conventions. Saves both PDF figure and NetCDF data file
# ===============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ===============================================================================
# 
# 6. plot_tc_duration_distribution(data_or_file, tdict)
#    Creates a histogram + KDE plot showing the distribution of TC lifetimes.
#    Displays mean and median durations with reference lines.
#
# 7. plot_tc_duration_by_category(data_or_file, tdict)
#    Produces a normalized histogram (PDF) of TC durations separated by 
#    Saffir-Simpson category. All curves are normalized to show relative
#    frequency distributions, making categories directly comparable.
#
# 8. plot_tc_basin_doughnut(data_or_file, tdict)
#    Creates a doughnut chart showing TC frequency by ocean basin following
#    Roberts et al. (2020) methodology. Includes comparison bar chart with
#    literature values. Accounts for cyclone season timing (NH: May-Nov,
#    SH: Oct-May).
#
# All statistical functions accept TCDataManager objects for fast in-memory
# processing and save publication-ready PDF figures.
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
from datetime import datetime
import xarray as xr


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
    
    Saves both PDF figure and NetCDF data file with trajectory information.
    
    Args:
        data_or_file: TCDataManager object OR file path string
        tdict: Configuration dictionary
        max_timesteps: Optional timestep limit
    """
    import xarray as xr
    
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
    
    # Collect data for NetCDF
    all_lons = []
    all_lats = []
    all_storm_ids = []
    all_timesteps = []
    
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
        
        # Collect for NetCDF
        all_lons.extend(lon_plot)
        all_lats.extend(lat)
        all_storm_ids.extend([storm_id] * len(lon_plot))
        all_timesteps.extend(range(len(lon_plot)))
    
    # Save figure
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    base_filename = f"tracks_{tdict['dataset']['model']}_{tdict['dataset']['exp']}_{startdate}_{enddate}"
    
    # Save PDF
    save_path_pdf = os.path.join(tdict['paths']['plotdir'], f"{base_filename}.pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=350)
    print(f"✓ PDF saved to: {save_path_pdf}")
    
    plt.show()
    plt.close()
    
    # =========================================================================
    # SAVE NETCDF
    # =========================================================================
    try:
        ds = xr.Dataset(
            {
                'longitude': (['obs'], all_lons),
                'latitude': (['obs'], all_lats),
                'storm_id': (['obs'], all_storm_ids),
                'timestep': (['obs'], all_timesteps),
            },
            coords={
                'obs': np.arange(len(all_lons))
            },
            attrs={
                'title': 'Tropical Cyclone Trajectories',
                'model': tdict['dataset']['model'],
                'experiment': tdict['dataset']['exp'],
                'start_date': startdate,
                'end_date': enddate,
                'n_storms': nstorms,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        
        save_path_nc = os.path.join(tdict['paths']['plotdir'], f"{base_filename}.nc")
        ds.to_netcdf(save_path_nc)
        print(f"✓ NetCDF saved to: {save_path_nc}")
        
    except Exception as e:
        print(f"⚠ Failed to save NetCDF: {e}")



def plot_trajectories_colored(data_or_file, tdict, color_by='category', category=None, max_timesteps=None):
    """
    Plot TC trajectories colored by category.
    
    This unified function can plot either:
    - ALL trajectories colored by intensity/category (if category=None)
    - ONLY trajectories of a specific category (if category=0-5)
    
    Trajectories are plotted from weakest to strongest, ensuring intense
    TCs (Cat 5) are always visible on top.
    
    Saves both PDF figure and NetCDF data file.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        color_by: 'intensity' (SLP) or 'category' (Saffir-Simpson) - only used if category=None
        category: If specified (0-5), plot only this category; if None, plot all
        max_timesteps: Optional timestep limit
    """
    
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    # =========================================================================
    # CATEGORY FILTERING (if requested)
    # =========================================================================
    if category is not None:
        # Filter mode: plot only one category
        cat_names = [
            'TD (≥1005 hPa)', 'Cat 1 (990–1004)', 'Cat 2 (975–989)',
            'Cat 3 (960–974)', 'Cat 4 (945–959)', 'Cat 5 (<945)'
        ]
        cat_colors = ['lightgreen', 'gold', 'orange', 'red', 'darkred', 'purple']
        
        print(f"Filtering for {cat_names[category]}...")
        
        filtered_trajectories = {}
        for storm_id, data in trajectories.items():
            min_slp = np.min(data['slp'])
            peak_category = category_from_slp_pa(min_slp)
            
            if peak_category == category:
                filtered_trajectories[storm_id] = data
        
        trajectories = filtered_trajectories
        nstorms = len(trajectories)
        
        if nstorms == 0:
            print(f"⚠ No trajectories found for {cat_names[category]}")
            return
        
        print(f"✓ Found {nstorms} trajectories for {cat_names[category]}")
        
        # Single color mode
        plot_color = cat_colors[category]
        use_colormap = False
        
    else:
        # All trajectories mode: use colormap
        nstorms = len(trajectories)
        use_colormap = True
        print(f"Plotting all {nstorms} storms colored by {color_by}")
    
    # =========================================================================
    # FIGURE SETUP
    # =========================================================================
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
    
    if use_colormap:
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        cmap = plt.cm.RdYlGn_r
        vmin, vmax = 0, 5
        label = 'Saffir–Simpson Category'
    
    # Colormap settings (only for all-trajectories mode)
    #if use_colormap:
        #if color_by == 'intensity':
        #    cmap = plt.cm.YlOrRd
        #    vmin, vmax = 920, 1010
        #    label = 'SLP (hPa)'
       # else:
       #     cmap = plt.cm.RdYlGn_r
       #     vmin, vmax = 0, 5
       #     label = 'Saffir–Simpson Category'
    
    # =========================================================================
    # SORT TRAJECTORIES BY INTENSITY (weak → strong for plotting order)
    # =========================================================================
    if use_colormap:
        storm_intensities = {}
        for storm_id, data in trajectories.items():
            min_slp = np.nanmin(data['slp'] / 100.0)
            storm_intensities[storm_id] = min_slp
        
        # Sort: weak (high SLP) first → strong (low SLP) last
        sorted_storm_ids = sorted(storm_intensities.keys(), 
                                 key=lambda sid: storm_intensities[sid], 
                                 reverse=True)  # High SLP first (weak)
        
        print(f"Plotting storms ordered: weak (background) → strong (foreground)")
    else:
        # For single category, no need to sort
        sorted_storm_ids = list(trajectories.keys())
    
    # Collect data for NetCDF
    nc_data = {
        'storm_id': [],
        'longitude': [],
        'latitude': [],
        'slp': [],
        'category': [],
        'timestep': []
    }
    
    # =========================================================================
    # PLOT TRAJECTORIES
    # =========================================================================
    for storm_id in sorted_storm_ids:
        data = trajectories[storm_id]
        
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
        
        if use_colormap:
            # LineCollection mode (colored by intensity/category)
            dlon = np.abs(np.diff(lon_plot))
            valid = dlon < 180
            
            min_slp = np.nanmin(slp)
            cat = category_from_slp_pa(min_slp * 100)
            values = np.full(len(lon_plot) - 1, cat)
            
            points = np.array([lon_plot, lat]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            segments = segments[valid]
            values = values[valid]
            
            linewidth = 0.8 + (5 - cat) * 0.15  # Cat 5 thicker
            zorder = 10 + (5 - cat)
            
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
                linewidth=linewidth,
                alpha=0.85,
                transform=ccrs.PlateCarree(),
                zorder=zorder
            )
            lc.set_array(values)
            ax.add_collection(lc)
            
        else:
            # Simple line plot mode (single category)
            lat_plot = lat.copy()
            
            # Handle dateline crossings
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
                color=plot_color,
                linewidth=1.5,
                alpha=0.7,
                transform=ccrs.PlateCarree()
            )
        
        # Collect for NetCDF
        for i in range(len(lon_plot)):
            nc_data['storm_id'].append(storm_id)
            nc_data['longitude'].append(lon_plot[i])
            nc_data['latitude'].append(lat[i])
            nc_data['slp'].append(slp[i])
            nc_data['category'].append(category_from_slp_pa(slp[i] * 100))
            nc_data['timestep'].append(i)
    
    # =========================================================================
    # COLORBAR (only for all-trajectories mode)
    # =========================================================================
    if use_colormap:
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(label, fontsize=12)
        
        if color_by == 'category':
            cbar.set_ticks([0, 1, 2, 3, 4, 5])
            cbar.set_ticklabels(['TD', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'])
    
    # =========================================================================
    # TITLE
    # =========================================================================
    if category is not None:
        title = (f"TC Tracks – {cat_names[category]}\n"
                f"{tdict['dataset']['model']} - {tdict['dataset']['exp']} "
                f"({nstorms} cyclones)")
    else:
        title = (f"TC Tracks coloured by {color_by}\n"
                f"{tdict['dataset']['model']} - {tdict['dataset']['exp']} "
                f"({nstorms} tracks)")
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # =========================================================================
    # SAVE FILES
    # =========================================================================
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    model_clean = tdict['dataset']['model'].replace(" ", "_")
    exp_clean = tdict['dataset']['exp'].replace(" ", "_")
    startdate_clean = startdate.replace(" ", "").replace("-", "")
    enddate_clean = enddate.replace(" ", "").replace("-", "")
    
    if category is not None:
        # Single category filename
        base_filename = f"tracks_cat{category}_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}"
    else:
        # All trajectories filename
        color_clean = str(color_by).replace(" ", "_")
        base_filename = f"tracks_colored_{color_clean}_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}"
    
    # Save PDF
    save_path_pdf = os.path.join(tdict['paths']['plotdir'], f"{base_filename}.pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=350)
    print(f"✓ PDF saved to: {save_path_pdf}")
    
    plt.show()
    plt.close()
    
    # =========================================================================
    # SAVE NETCDF
    # =========================================================================
    try:
        if category is not None:
            title_nc = f'Tropical Cyclone Trajectories - Category {category}'
            category_info = category
        else:
            title_nc = f'Tropical Cyclone Trajectories colored by {color_by}'
            category_info = 'all'
        
        ds = xr.Dataset(
            {
                'longitude': (['obs'], nc_data['longitude']),
                'latitude': (['obs'], nc_data['latitude']),
                'slp': (['obs'], nc_data['slp']),
                'category': (['obs'], nc_data['category']),
                'storm_id': (['obs'], nc_data['storm_id']),
                'timestep': (['obs'], nc_data['timestep']),
            },
            coords={
                'obs': np.arange(len(nc_data['longitude']))
            },
            attrs={
                'title': title_nc,
                'model': tdict['dataset']['model'],
                'experiment': tdict['dataset']['exp'],
                'start_date': startdate,
                'end_date': enddate,
                'n_storms': nstorms,
                'filter_category': str(category_info),
                'color_by': color_by if category is None else 'single_category',
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        
        save_path_nc = os.path.join(tdict['paths']['plotdir'], f"{base_filename}.nc")
        ds.to_netcdf(save_path_nc)
        print(f"✓ NetCDF saved to: {save_path_nc}")
        
    except Exception as e:
        print(f"⚠ Failed to save NetCDF: {e}")




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
    
    Saves both PDF figure and NetCDF data file with gridded density information.
    The colorbar range adapts automatically to the actual data range.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        grid_size: Grid cell size in degrees (default 2.5)
        max_timesteps: Optional timestep limit
    """
    import xarray as xr
    
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
    
    # Data range
    vmax_data = transits_per_month.max()
    vmin_data = transits_per_month[transits_per_month > 0].min() if np.any(transits_per_month > 0) else 0.01
    
    print(f"Data range: {vmin_data:.4f} - {vmax_data:.2f} transits/month")
    
    # =========================================================================
    # ADAPTIVE COLORBAR BOUNDARIES
    # =========================================================================
    # Use actual max as upper bound (rounded up nicely)
    if vmax_data < 0.5:
        vmax_plot = np.ceil(vmax_data * 20) / 20  # Round to nearest 0.05
    elif vmax_data < 1.0:
        vmax_plot = np.ceil(vmax_data * 10) / 10  # Round to nearest 0.1
    elif vmax_data < 3.0:
        vmax_plot = np.ceil(vmax_data * 2) / 2    # Round to nearest 0.5
    else:
        vmax_plot = np.ceil(vmax_data)             # Round to nearest 1
    
    vmin_plot = max(0.01, vmin_data)
    
    print(f"Colorbar range: {vmin_plot:.4f} - {vmax_plot:.2f}")
    
    # Create adaptive boundaries (12 levels)
    n_levels = 12
    boundaries = np.logspace(
        np.log10(vmin_plot), 
        np.log10(vmax_plot), 
        n_levels + 1
    )
    boundaries[0] = 0.0  # Start from zero
    
    print(f"Colorbar boundaries: {boundaries}")
    
    # Discretize
    transits_discrete = np.digitize(transits_per_month, boundaries) - 1
    transits_discrete = np.clip(transits_discrete, 0, n_levels - 1)
    transits_discrete_masked = np.where(transits_per_month > 0, transits_discrete, np.nan)
    
    # Colormap
    base_colors = ['#FFFFFF', '#8B4513', '#D2691E', '#FFD700', '#ADFF2F', '#00FF00', '#00CED1', '#0000FF']
    cmap_continuous = LinearSegmentedColormap.from_list('tc_density', base_colors, N=256)
    colors_discrete = [cmap_continuous(i / n_levels) for i in range(n_levels)]
    cmap_discrete = ListedColormap(colors_discrete)
    
    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
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
    
    # =========================================================================
    # ADAPTIVE COLORBAR
    # =========================================================================
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
    
    # Create adaptive tick labels
    tick_labels = []
    for i in range(n_levels):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == n_levels - 1:
            tick_labels.append(f'>{lower:.2g}')
        else:
            # Adaptive formatting based on value range
            if upper < 0.1:
                tick_labels.append(f'{lower:.3f}–{upper:.3f}')
            elif upper < 1.0:
                tick_labels.append(f'{lower:.2f}–{upper:.2f}')
            else:
                tick_labels.append(f'{lower:.2g}–{upper:.2g}')
    
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8, rotation=45)
    
    # =========================================================================
    # TITLE
    # =========================================================================
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
    
    base_filename = f'track_density_grid_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}'
    
    # =========================================================================
    # SAVE PDF
    # =========================================================================
    save_path_pdf = os.path.join(tdict['paths']['plotdir'], f'{base_filename}.pdf')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ PDF saved to: {save_path_pdf}")
    
    plt.show()
    plt.close()
    
    # =========================================================================
    # SAVE NETCDF
    # =========================================================================
    try:
        # Grid centers for coordinate arrays
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        
        ds = xr.Dataset(
            {
                'transits_per_month': (['latitude', 'longitude'], transits_per_month.T),
                'count': (['latitude', 'longitude'], counts.T),
                'transits_discrete': (['latitude', 'longitude'], transits_discrete_masked.T),
            },
            coords={
                'longitude': lon_centers,
                'latitude': lat_centers,
            },
            attrs={
                'title': 'TC Track Density Grid',
                'description': 'Gridded tropical cyclone track density expressed as transits per month',
                'model': model,
                'experiment': exp,
                'start_date': startdate,
                'end_date': enddate,
                'n_months': n_months,
                'grid_size_degrees': grid_size,
                'total_observations': total_points,
                'max_transits_per_month': float(vmax_data),
                'min_transits_per_month': float(vmin_data),
                'colorbar_min': float(vmin_plot),
                'colorbar_max': float(vmax_plot),
                'n_discrete_levels': n_levels,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'units_transits_per_month': 'count per month',
                'units_count': 'total count'
            }
        )
        
        # Add variable attributes
        ds['transits_per_month'].attrs = {
            'long_name': 'TC track density',
            'units': 'transits per month',
            'description': 'Number of TC observations per grid cell normalized by time period'
        }
        
        ds['count'].attrs = {
            'long_name': 'TC observation count',
            'units': 'count',
            'description': 'Total number of TC observations in each grid cell'
        }
        
        ds['transits_discrete'].attrs = {
            'long_name': 'Discretized track density',
            'units': 'level index',
            'description': f'Transits per month discretized into {n_levels} levels for plotting'
        }
        
        save_path_nc = os.path.join(tdict['paths']['plotdir'], f'{base_filename}.nc')
        ds.to_netcdf(save_path_nc)
        print(f"✓ NetCDF saved to: {save_path_nc}")
        
        # Print NetCDF info
        print(f"  Grid dimensions: {len(lat_centers)} lat × {len(lon_centers)} lon")
        print(f"  Data variables: transits_per_month, count, transits_discrete")
        
    except Exception as e:
        print(f"⚠ Failed to save NetCDF: {e}")
        import traceback
        traceback.print_exc()

        

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





def plot_tc_duration_distribution(data_or_file, tdict):
    """
    Plot TC lifetime distribution with histogram and KDE.
    
    Creates a histogram + kernel density estimate showing the distribution
    of tropical cyclone lifetimes, with mean and median reference lines.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
    """
    from scipy.stats import gaussian_kde
    
    print("Computing TC duration statistics from memory...")
    
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    durations_days = []
    
    for storm_id, storm_data in trajectories.items():
        years = storm_data['year']
        months = storm_data['month']
        days = storm_data['day']
        hours = storm_data['hour']
        
        try:
            start_dt = datetime(years[0], months[0], days[0], hours[0])
            end_dt = datetime(years[-1], months[-1], days[-1], hours[-1])
            delta = end_dt - start_dt
            duration_days = delta.total_seconds() / (24 * 3600)
            durations_days.append(duration_days)
        except:
            continue
    
    durations_days = np.array(durations_days)
    
    # Statistics
    print("\n" + "="*70)
    print("TC DURATION STATISTICS")
    print("="*70)
    print(f"Total cyclones: {len(durations_days)}")
    print(f"Mean duration: {durations_days.mean():.1f} days")
    print(f"Median duration: {np.median(durations_days):.1f} days")
    print(f"Min duration: {durations_days.min():.1f} days ({durations_days.min()*24:.0f} hours)")
    print(f"Max duration: {durations_days.max():.1f} days")
    print(f"Std deviation: {durations_days.std():.1f} days")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    n, bins, patches = ax.hist(durations_days, bins=40, 
                               color='steelblue', alpha=0.7, 
                               edgecolor='black', linewidth=1.2,
                               density=True, label='Histogram')
    
    # KDE
    kde = gaussian_kde(durations_days)
    x_range = np.linspace(durations_days.min(), durations_days.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
    
    # Reference lines
    ax.axvline(durations_days.mean(), color='green', linestyle='--', 
              linewidth=2, label=f'Mean: {durations_days.mean():.1f} days')
    ax.axvline(np.median(durations_days), color='orange', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(durations_days):.1f} days')
    
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    ax.set_xlabel('Duration (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title(f'TC Lifetime Distribution\n'
                 f'{startdate} – {enddate} | Dataset: {tdict["dataset"]["model"]}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'tc_duration_distribution_{tdict["dataset"]["model"]}_{tdict["dataset"]["exp"]}_{startdate}_{enddate}.pdf'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Duration distribution saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_tc_duration_by_category(data_or_file, tdict):
    """
    Plot normalized TC duration histograms by Saffir-Simpson category.
    
    Creates overlaid normalized histograms (PDFs) showing duration distributions
    for each intensity category. Normalization allows direct comparison between
    categories regardless of sample size.
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
    """
    print("Computing TC duration by category from memory...")
    
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    durations_by_cat = {cat: [] for cat in range(6)}
    
    for storm_id, storm_data in trajectories.items():
        years = storm_data['year']
        months = storm_data['month']
        days = storm_data['day']
        hours = storm_data['hour']
        slps = storm_data['slp']
        
        try:
            start_dt = datetime(years[0], months[0], days[0], hours[0])
            end_dt = datetime(years[-1], months[-1], days[-1], hours[-1])
            delta = end_dt - start_dt
            duration_days = delta.total_seconds() / (24 * 3600)
        except:
            continue
        
        max_cat = max([category_from_slp_pa(slp) for slp in slps])
        durations_by_cat[max_cat].append(duration_days)
    
    # Statistics
    print("\nTC count per category:")
    for cat in range(6):
        n = len(durations_by_cat[cat])
        if n > 0:
            mean_dur = np.mean(durations_by_cat[cat])
            print(f"  Cat {cat}: {n} storms, mean duration: {mean_dur:.1f} days")
    
    # Plot
    cat_names = ['TD', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
    colors = ['#1b9e77', '#d95f02', '#e6ab02', '#e7298a', '#a6761d', '#7570b3']
    
    max_duration = max(max(v) for v in durations_by_cat.values() if len(v) > 0)
    bins = np.arange(0, int(np.ceil(max_duration)) + 2, 1)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for cat in range(6):
        data = durations_by_cat.get(cat, [])
        if len(data) == 0:
            continue
        
        # NORMALIZED histogram (density=True creates PDF)
        ax.hist(
            data,
            bins=bins,
            histtype='step',
            linewidth=2.2,
            color=colors[cat],
            label=f'{cat_names[cat]} (n={len(data)})',
            density=True,  # ← NORMALIZATION: makes it a PDF
            alpha=0.85
        )
    
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    ax.set_xlabel('TC duration (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (probability density)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Normalized TC Duration Distribution by Category\n'
        f'{startdate} – {enddate}',
        fontsize=13,
        fontweight='bold'
    )
    
    ax.grid(alpha=0.25, linestyle=':')
    ax.legend(fontsize=9, ncol=2, frameon=False)
    
    plt.tight_layout()
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'tc_duration_normalized_by_category_{tdict["dataset"]["model"]}_{tdict["dataset"]["exp"]}_{startdate}_{enddate}.pdf'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Normalized duration histogram saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_tc_basin_doughnut(data_or_file, tdict, reference_freq=90.0):
    """
    Plot TC frequency doughnut chart by ocean basin with comparison.
    
    Creates a doughnut chart showing annual TC frequency for each ocean basin
    following Roberts et al. (2020) methodology. Includes:
    - Doughnut thickness scaled to reference climatology
    - Separate NH/SH totals in center
    - Comparison bar chart with Roberts et al. (2020) values
    - Basin classification using IBTrACS standards
    - Cyclone season filtering (NH: May-Nov, SH: Oct-May)
    
    Args:
        data_or_file: TCDataManager object OR file path
        tdict: Configuration dictionary
        reference_freq: Reference NH frequency for scaling (default 90.0 storms/year)
    """
    print("Computing basin frequencies from memory...")
    
    trajectories, data_dict = _get_data_from_input(data_or_file)
    
    # Cyclone seasons
    nh_season_months = [5, 6, 7, 8, 9, 10, 11]
    sh_season_months = [10, 11, 12, 1, 2, 3, 4, 5]
    
    nh_basins = ['North Atlantic', 'East Pacific', 'West Pacific', 'North Indian']
    sh_basins = ['South Indian', 'South Pacific']
    
    basin_counts = {basin: set() for basin in nh_basins + sh_basins}
    years_tracked = set()
    
    for storm_id, storm_data in trajectories.items():
        year = storm_data['year'][0]
        month = storm_data['month'][0]
        lon = storm_data['lon'][0]
        lat = storm_data['lat'][0]
        
        years_tracked.add(year)
        
        basin = get_basin_ibtracs(lon, lat)
        hemisphere = 'NH' if lat >= 0 else 'SH'
        
        is_nh_season = month in nh_season_months
        is_sh_season = month in sh_season_months
        
        if hemisphere == 'NH' and is_nh_season and basin in nh_basins:
            basin_counts[basin].add(storm_id)
        elif hemisphere == 'SH' and is_sh_season and basin in sh_basins:
            basin_counts[basin].add(storm_id)
    
    n_years = len(years_tracked)
    print(f"Period: {min(years_tracked)}-{max(years_tracked)} ({n_years} years)")
    
    basin_freq = {basin: len(storms) / n_years for basin, storms in basin_counts.items()}
    
    nh_total = sum([basin_freq[b] for b in nh_basins])
    sh_total = sum([basin_freq[b] for b in sh_basins])
    
    print(f"\nAnnual frequencies (storms/year):")
    print(f"  NH total: {nh_total:.1f}")
    print(f"  SH total: {sh_total:.1f}")
    for basin in nh_basins + sh_basins:
        print(f"    {basin}: {basin_freq[basin]:.1f}")
    
    # =========================================================================
    # DOUGHNUT CHART (COMPACT)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 9))  # Slightly smaller
    
    colors = {
        'North Atlantic': '#3498db',
        'East Pacific': '#2ecc71',
        'West Pacific': '#e74c3c',
        'North Indian': '#f39c12',
        'South Indian': '#9b59b6',
        'South Pacific': '#34495e'
    }
    
    basins_ordered = ['North Atlantic', 'East Pacific', 'West Pacific', 
                      'North Indian', 'South Indian', 'South Pacific']
    
    sizes = [basin_freq[b] for b in basins_ordered]
    colors_list = [colors[b] for b in basins_ordered]
    
    scale_factor = nh_total / reference_freq
    
    # COMPACT: smaller radii
    inner_radius = 0.20
    outer_radius = 0.20 + (0.28 * scale_factor)
    
    global_total = sum(sizes)
    basin_percentages = [100.0 * basin_freq[b] / global_total for b in basins_ordered]
    
    wedges, texts = ax.pie(sizes, 
                           colors=colors_list,
                           startangle=90,
                           counterclock=False,
                           radius=outer_radius,
                           wedgeprops=dict(width=outer_radius-inner_radius, 
                                          edgecolor='white', 
                                          linewidth=2))
    
    # Percentages on wedges
    for wedge, pct in zip(wedges, basin_percentages):
        if pct < 3.0:
            continue
        
        angle = 0.5 * (wedge.theta1 + wedge.theta2)
        angle_rad = np.deg2rad(angle)
        r_text = inner_radius + 0.5 * (outer_radius - inner_radius)
        x = r_text * np.cos(angle_rad)
        y = r_text * np.sin(angle_rad)
        
        ax.text(x, y, f'{pct:.0f}%',
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white', zorder=12)
    
    # Inner circle
    from matplotlib.patches import Circle
    circle = Circle((0, 0), inner_radius, color='white', zorder=10)
    ax.add_artist(circle)
    
    # Center text - LARGER NH/SH labels
    ax.text(0, 0.08, f'{nh_total:.1f}', 
           ha='center', va='center', fontsize=32, fontweight='bold',
           color='black', zorder=11)
    ax.text(0, -0.08, f'{sh_total:.1f}', 
           ha='center', va='center', fontsize=28, fontweight='bold',
           color='gray', zorder=11)
    ax.text(0, 0.15, 'NH', ha='center', va='center', fontsize=20, 
           fontweight='bold', color='black', zorder=11)
    ax.text(0, -0.15, 'SH', ha='center', va='center', fontsize=20, 
           fontweight='bold', color='gray', zorder=11)
    
    # Title
    ax.set_title(f'Tropical Cyclone Frequency by Basin\n'
                 f'(mean storms per year, {min(years_tracked)}–{max(years_tracked)})\n'
                 f'NH: May–Nov | SH: Oct–May',
                 fontsize=13, fontweight='bold', pad=15)
    
    # Legend
    legend_labels = [f'{basin}: {basin_freq[basin]:.1f}/yr' for basin in basins_ordered]
    ax.legend(wedges, legend_labels, 
             loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
             fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # NO caption/note (removed as requested)
    ax.axis('equal')
    
    plt.tight_layout()
    
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'tc_doughnut_frequency_{tdict["dataset"]["model"]}_{tdict["dataset"]["exp"]}_{startdate}_{enddate}.pdf'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Doughnut chart saved: {save_path}")
    
    plt.show()
    plt.close()
    
    # =========================================================================
    # COMPARISON BAR CHART
    # =========================================================================
    roberts_era5 = {
        'North Atlantic': 12.5, 'East Pacific': 16.0, 'West Pacific': 25.0,
        'North Indian': 5.5, 'South Indian': 10.0, 'South Pacific': 4.5,
        'NH_total': 59.0, 'SH_total': 14.5
    }
    
    print("\n" + "="*70)
    print("COMPARISON WITH ROBERTS ET AL. (2020)")
    print("="*70)
    print(f"{'Basin':<20} {'This study':<18} {'Roberts et al.':<18} {'Difference':<15}")
    print("-"*70)
    
    for basin in basins_ordered:
        our_val = basin_freq[basin]
        roberts_val = roberts_era5.get(basin, np.nan)
        diff = our_val - roberts_val
        diff_pct = (diff / roberts_val * 100) if roberts_val > 0 else 0
        print(f"{basin:<20} {our_val:>16.1f} {roberts_val:>16.1f} {diff:>8.1f} ({diff_pct:+.0f}%)")
    
    print("-"*70)
    print(f"{'NH Total':<20} {nh_total:>16.1f} {roberts_era5['NH_total']:>16.1f} "
          f"{nh_total - roberts_era5['NH_total']:>8.1f} "
          f"({(nh_total - roberts_era5['NH_total'])/roberts_era5['NH_total']*100:+.0f}%)")
    print(f"{'SH Total':<20} {sh_total:>16.1f} {roberts_era5['SH_total']:>16.1f} "
          f"{sh_total - roberts_era5['SH_total']:>8.1f} "
          f"({(sh_total - roberts_era5['SH_total'])/roberts_era5['SH_total']*100:+.0f}%)")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(basins_ordered))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [basin_freq[b] for b in basins_ordered], width,
                  label='This study', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, [roberts_era5[b] for b in basins_ordered], width,
                  label='Roberts et al. (2020)', color='coral', edgecolor='black')
    
    ax.set_xlabel('Ocean Basin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Storms per year', fontsize=12, fontweight='bold')
    ax.set_title(f'TC Frequency Comparison: This Study vs Roberts et al. (2020)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace(' ', '\n') for b in basins_ordered], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path_comp = os.path.join(
        tdict['paths']['plotdir'],
        f'tc_frequency_comparison_{tdict["dataset"]["model"]}_{tdict["dataset"]["exp"]}_{startdate}_{enddate}.pdf'
    )
    plt.savefig(save_path_comp, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved: {save_path_comp}")
    
    plt.show()
    plt.close()
    
    print("✓ Basin analysis complete!")    