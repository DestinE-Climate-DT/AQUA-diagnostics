# ===========================================
# MODIFIED PLOTTING FUNCTIONS FOR DIRECT FORMAT
# ===========================================
# plotting_TCs_custom.py
#
# This module provides a set of utilities to read, classify,
# and visualize tropical cyclone (TC) trajectories directly
# from the filtered trajectory file format (no intermediate
# conversion required).
#
# The main functionalities included are:
#
# 1) category_from_slp_pa(slp_pa)
#    --------------------------------------------------------
#    Assigns a Saffir–Simpson category based on minimum
#    sea-level pressure (SLP, in Pa).
#    Categories:
#        0 = Tropical Depression (TD)
#        1–5 = Hurricane Categories 1 to 5
#
# 1b) get_basin_ibtracs(lon,lat)
#
# 2) getTrajectories_direct(filename)
#    --------------------------------------------------------
#    Reads TC trajectories directly from the filtered file.
#    Returns:
#        - number of trajectories
#        - maximum trajectory length
#        - dictionary with storm-wise data (lon, lat, slp, wind, time)
#
# 3) plot_trajectories_direct(trajfile, tdict, max_timesteps=None)
#    --------------------------------------------------------
#    Plots all TC trajectories as simple scatter points
#    (black dots), without intensity or category information.
#    Useful for quick diagnostics and sanity checks.
#
# 4) plot_trajectories_colored(trajfile, tdict,
#                              color_by='intensity' or 'category',
#                              max_timesteps=None)
#    --------------------------------------------------------
#    Plots TC trajectories as continuous lines using
#    LineCollection, with:
#        - one color per cyclone
#        - color based on minimum SLP (intensity) or
#          Saffir–Simpson category
#    Handles dateline crossings to avoid spurious long lines.
#    Includes a consistent colorbar.
#
# 5) plot_trajectories_by_category(trajfile, tdict,
#                                  category=0–5,
#                                  max_timesteps=None)
#    --------------------------------------------------------
#    Plots ONLY tropical cyclones belonging to a single
#    Saffir–Simpson category, using continuous colored lines.
#    The category is determined from the minimum SLP reached
#    during the cyclone lifetime.
#
#
# ==============================================================================
# Density-based visualizations of tropical cyclone (TC) tracks
# ==============================================================================
#
# This module provides two complementary approaches to visualize the spatial
# density of tropical cyclone (TC) tracks:
#
# 6) plot_density_scatter
#    --------------------------------------------------------------------------
#    Produces a point-based density scatter plot where each individual TC
#    observation (lon/lat) is colored according to its local spatial density,
#    estimated using Gaussian Kernel Density Estimation (KDE).
#
#    Key characteristics:
#      - All TC track points are extracted from the filtered trajectory file
#      - Local density is computed using scipy.stats.gaussian_kde
#      - Density values are normalized to the [0, 1] range
#      - Points are sorted by density before plotting:
#            * low-density points are plotted first (background)
#            * high-density points are plotted last (foreground)
#        ensuring that hotspots remain visible and are not obscured
#      - For performance, the KDE is computed on a random subsample
#        (default: 50,000 points) if the dataset is larger
#      - A horizontal colorbar is used, following HighResMIP-PRIMAVERA style
#
#    This representation emphasizes fine-scale spatial structures and
#    overlapping tracks, making it particularly suitable for identifying
#    preferred genesis regions and major TC corridors.
#
#
# 7) plot_track_density_grid
#    --------------------------------------------------------------------------
#    Produces a gridded TC track density map expressed as "transits per month",
#    following the HighResMIP-PRIMAVERA methodology (e.g. Page 10, Fig. 3).
#
#    Key characteristics:
#      - The domain is divided into regular latitude–longitude grid cells
#        (default resolution: 2.5° × 2.5°)
#      - TC observations are counted within each grid cell
#      - Counts are normalized by the length of the analysis period
#        to obtain transits per month
#      - A discrete, logarithmically spaced color scale is used to improve
#        interpretability across orders of magnitude
#      - Zero-density grid cells are masked
#      - A custom HighResMIP-style colormap is applied
#        (white → brown → yellow → green → blue)
#      - The colorbar is horizontal and uses labeled discrete intervals
#
#    This representation provides a more aggregated, statistically robust view
#    of TC activity, facilitating quantitative comparisons across models,
#    experiments, and time periods.
#
# Together, these two functions (6-7) offer complementary perspectives:
#   - the density scatter plot highlights fine-scale clustering and overlap
#   - the gridded density map emphasizes large-scale, climatological patterns
#
# 
#
# 8) plot_density_scatter_by_category()
#    6-panel subplot showing KDE density separately for each Saffir-Simpson
#    category (TD, Cat 1-5). Reveals how spatial patterns vary with intensity.
#

# All plotting functions use Cartopy with a PlateCarree
# projection and are designed to produce publication-ready
# figures (PDF output).
# ============================================================


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import os
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm,  ListedColormap, LogNorm
from scipy.stats import gaussian_kde
from datetime import datetime



# ===== def category
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
# ========================================



# ==== def basins
def get_basin_ibtracs(lon, lat):
    """
    Classification based on IBTrACS (WMO standard)
    Ref: https://www.ncei.noaa.gov/products/international-best-track-archive
    """
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
# ==============================================

def getTrajectories_direct(filename):
    """
    Read trajectories directly from the filtered_file format without conversion.
    
    Returns:
        numtraj: Total number of trajectories
        maxNumPts: Maximum length of a trajectory
        trajectories: Dictionary containing storm-wise data
    """
    print(f"Reading trajectories directly from: {filename}")
    
    trajectories = {}
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip header if present
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
            
            # Initialize storm if it does not exist
            if storm_id not in trajectories:
                trajectories[storm_id] = {
                    'lon': [],
                    'lat': [],
                    'slp': [],
                    'wind': [],
                    'year': [],
                    'month': []
                }
            
            # Append track point
            trajectories[storm_id]['lon'].append(lon)
            trajectories[storm_id]['lat'].append(lat)
            trajectories[storm_id]['slp'].append(slp)
            trajectories[storm_id]['wind'].append(wind)
            trajectories[storm_id]['year'].append(year)
            trajectories[storm_id]['month'].append(month)
    
    # Convert lists to numpy arrays
    for storm_id in trajectories:
        for key in trajectories[storm_id]:
            trajectories[storm_id][key] = np.array(trajectories[storm_id][key])
    
    numtraj = len(trajectories)
    maxNumPts = max([len(t['lon']) for t in trajectories.values()])
    
    print(f"Found {numtraj} trajectories")
    print(f"Maximum number of points per trajectory: {maxNumPts}")
    
    return numtraj, maxNumPts, trajectories


def plot_trajectories_direct(trajfile, tdict, max_timesteps=None):
    """
    Plot trajectories directly from the filtered_file format.
    
    Args:
        trajfile: Path to the filtered file
        tdict: Configuration dictionary
        max_timesteps: Optional timestep limit for plotting (None = no limit)
    """
    print(f"Plotting trajectories from: {trajfile}")
    
    # Read trajectories
    nstorms, max_pts, trajectories = getTrajectories_direct(trajfile)
    
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
        
        # Limit timesteps if requested
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
        
        # Convert longitude from [0, 360] to [-180, 180]
        lon_plot = np.where(lon > 180, lon - 360, lon)
        
        # Scatter plot
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
    enddate   = tdict['time']['enddate']

    save_path = os.path.join(
      tdict['paths']['plotdir'],
      f"tracks_{tdict['dataset']['model']}_{tdict['dataset']['exp']}_{startdate}_{enddate}.pdf"
    )

    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close()


def plot_trajectories_colored(trajfile, tdict, color_by='intensity', max_timesteps=None):
    """
    Plot trajectories colored by intensity or category.
    
    Args:
        trajfile: Path to the filtered file
        tdict: Configuration dictionary
        color_by: 'intensity' (SLP) or 'category' (Saffir–Simpson)
        max_timesteps: Optional timestep limit for plotting
    """
    print(f"Plotting colored trajectories from: {trajfile}")
    
    # Read trajectories
    nstorms, max_pts, trajectories = getTrajectories_direct(trajfile)
    
    # Initialize figure
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
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
    
    # Colormap settings
    if color_by == 'intensity':
        cmap = plt.cm.YlOrRd
        vmin, vmax = 920, 1010  # hPa
        label = 'SLP (hPa)'
    else:  # category
        cmap = plt.cm.RdYlGn_r
        vmin, vmax = 0, 5
        label = 'Saffir–Simpson Category'
    
    # Plot each trajectory
    all_values = []
    for storm_id, data in trajectories.items():

        lon = data['lon']
        lat = data['lat']
        slp = data['slp'] / 100.0  # Pa → hPa

        # Limit timesteps if requested
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
            slp = slp[:max_timesteps]

        # Convert longitude
        lon_plot = np.where(lon > 180, lon - 360, lon)

        # Break tracks at dateline crossings (±180°)
        dlon = np.abs(np.diff(lon_plot))
        valid = dlon < 180  # True = safe segment

        # Skip too-short tracks
        if len(lon_plot) < 2:
            continue

        # ==================================================
        # ONE color per cyclone (based on MIN SLP)
        # ==================================================
        min_slp = np.nanmin(slp)          # hPa
        cat = category_from_slp_pa(min_slp * 100)

        values = np.full(len(lon_plot) - 1, cat)

        # ==================================================
        # Build line segments
        # ==================================================
        points = np.array([lon_plot, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        segments = segments[valid]
        values   = values[valid]
        # ==================================================
        # LineCollection
        # ==================================================
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


    

    sm = ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])

    # colour bar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    #cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=12)
    
    if color_by == 'slp category':
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['TD', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'])

    # Title
    plt.title(
        f"TC Tracks coloured by {color_by}\n"
        f"{tdict['dataset']['model']} - {tdict['dataset']['exp']} "
        f"({nstorms} tracks)",
        fontsize=14,
        fontweight='bold'
    )
    
    # Save figure
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    startdate = tdict['time']['startdate']
    enddate   = tdict['time']['enddate']

    # Avoid spaces in the name of the figure pdf
    model_clean = tdict['dataset']['model'].replace(" ", "_")
    exp_clean = tdict['dataset']['exp'].replace(" ", "_")
    color_clean = str(color_by).replace(" ", "_")  # se vuoi essere sicuro

    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f"tracks_colored_{color_clean}_{model_clean}_{exp_clean}_{startdate}_{enddate}.pdf"
    )

    plt.savefig(save_path, bbox_inches='tight', dpi=350)

    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close()



def plot_trajectories_by_category(trajfile, tdict, category=1, max_timesteps=None):
    """
    Plot ONLY tropical cyclones belonging to a specific category using continuous lines.
    
    Args:
        trajfile: Path to the filtered trajectory file
        tdict: Configuration dictionary
        category: Category to plot (0 = TD, 1–5 = Cat 1–5)
        max_timesteps: Timestep limit (None = all timesteps)
    """
    cat_names = [
        'TD (≥1005 hPa)',
        'Cat 1 (990–1004)',
        'Cat 2 (975–989)',
        'Cat 3 (960–974)',
        'Cat 4 (945–959)',
        'Cat 5 (<945)'
    ]
    cat_colors = ['lightgreen', 'gold', 'orange', 'red', 'darkred', 'purple']
    
    print(f"Plotting trajectories for {cat_names[category]}...")
    
    # Read data
    nstorms, max_pts, trajectories = getTrajectories_direct(trajfile)
    
    # Filter only cyclones of the requested category (based on minimum SLP)
    filtered_trajectories = {}
    for storm_id, data in trajectories.items():
        slp = data['slp']
        min_slp = np.min(slp)
        peak_category = category_from_slp_pa(min_slp)
        
        if peak_category == category:
            filtered_trajectories[storm_id] = data
    
    n_filtered = len(filtered_trajectories)
    
    if n_filtered == 0:
        print(f"⚠ No trajectories found for {cat_names[category]}")
        return
    
    print(f"✓ Found {n_filtered} trajectories for {cat_names[category]}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
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
    
    # Plot each trajectory
    for storm_id, data in filtered_trajectories.items():
        lon = data['lon']
        lat = data['lat']
        
        # Limit timesteps if requested
        if max_timesteps is not None:
            lon = lon[:max_timesteps]
            lat = lat[:max_timesteps]
        
        # Convert longitude
        lon_plot = np.where(lon > 180, lon - 360, lon)
        lat_plot = lat.copy()

        # Avoid lon ±180°
        dlon = np.diff(lon_plot)
        split_indices = np.where(np.abs(dlon) > 180)[0]

        # Insert NaN where lon crosses +179 <--> -179
        lon_plot_fixed = lon_plot.copy()
        lat_plot_fixed = lat_plot.copy()
        for idx in split_indices:
            lon_plot_fixed[idx+1] = np.nan
            lat_plot_fixed[idx+1] = np.nan

        # Plot
        ax.plot(
            lon_plot_fixed,
            lat_plot_fixed,
            color=cat_colors[category],
            linewidth=1.5,
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )

    # Title
    plt.title(
        f"TC Tracks – {cat_names[category]}\n"
        f"{tdict['dataset']['model']} - {tdict['dataset']['exp']} "
        f"({n_filtered} tropical cyclones)",
        fontsize=14,
        fontweight='bold'
    )
    
    # Save figure
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)

    # Replace spaces from the name of the figure to save it
    model_clean = tdict['dataset']['model'].replace(" ", "_")
    exp_clean = tdict['dataset']['exp'].replace(" ", "_")
    
    # FIX: Use tdict['time'] instead of tdict['dataset']
    startdate_clean = tdict['time']['startdate'].replace(" ", "").replace("-", "")
    enddate_clean = tdict['time']['enddate'].replace(" ", "").replace("-", "")

    # Save the figure with selected start/end date
    save_filename = (
        f"tracks_cat{category}_"
        f"{model_clean}_"
        f"{exp_clean}_"
        f"{startdate_clean}_"
        f"{enddate_clean}.pdf"
    )

    save_path = os.path.join(tdict['paths']['plotdir'], save_filename)

    plt.savefig(save_path, bbox_inches='tight', dpi=350)

    print(f"✓ Plot saved: {save_path}")
    plt.show()
    plt.close() 



# density scatter plots:
#
def plot_density_scatter(trajfile, tdict, max_timesteps=None, sample_size=50000):
    """
    Plot TC track density scatter with KDE-based coloring and point sorting.
    
    Points are sorted by density so that:
    - Low-density points are plotted FIRST (bottom layer)
    - High-density points are plotted LAST (top layer)
    This improves visualization by ensuring hotspots are visible on top.
    
    Args:
        trajfile: Path to the filtered trajectory file
        tdict: Configuration dictionary with dataset info and paths
        max_timesteps: Optional limit on trajectory length (None = all points)
        sample_size: Maximum number of points for KDE computation (default 50000)
    
    Returns:
        None (saves plot to file)
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import os
    
    print(f"Creating density scatter plot from: {trajfile}")
    
    # =========================================================================
    # 1. READ TRAJECTORIES AND EXTRACT ALL POINTS
    # =========================================================================
    lon_all = []
    lat_all = []
    
    with open(trajfile, 'r') as f:
        for line in f:
            # Skip header
            if 'track_id' in line or 'year' in line:
                continue
            
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            
            lon = float(parts[7])
            lat = float(parts[8])
            
            # Convert longitude from [0, 360] to [-180, 180]
            if lon > 180:
                lon -= 360
            
            lon_all.append(lon)
            lat_all.append(lat)
    
    lon_all = np.array(lon_all)
    lat_all = np.array(lat_all)
    
    total_points = len(lon_all)
    print(f"Total TC observation points: {total_points:,}")
    
    # =========================================================================
    # 2. SAMPLE FOR PERFORMANCE IF NEEDED
    # =========================================================================
    if total_points > sample_size:
        print(f"Sampling {sample_size:,} points for KDE computation...")
        indices = np.random.choice(total_points, sample_size, replace=False)
        lon_sample = lon_all[indices]
        lat_sample = lat_all[indices]
    else:
        lon_sample = lon_all
        lat_sample = lat_all
    
    # =========================================================================
    # 3. COMPUTE LOCAL DENSITY USING KDE
    # =========================================================================
    print("Computing Kernel Density Estimation...")
    try:
        xy = np.vstack([lon_sample, lat_sample])
        kde = gaussian_kde(xy)
        density = kde(xy)
        
        # Normalize to [0, 1] scale
        density_normalized = density / density.max()
        
        print(f"Density range: {density.min():.6f} - {density.max():.6f}")
        print(f"KDE bandwidth: {kde.factor:.4f}")
        
    except Exception as e:
        print(f"KDE computation failed: {e}")
        print("Falling back to uniform coloring")
        density_normalized = np.ones(len(lon_sample))
    
    # =========================================================================
    # 4. SORT POINTS BY DENSITY (Low to High)
    # =========================================================================
    # This ensures high-density points are plotted on top
    print("Sorting points by density for better visualization...")
    idx = density_normalized.argsort()  # Indices that would sort the array
    
    lon_sorted = lon_sample[idx]
    lat_sorted = lat_sample[idx]
    density_sorted = density_normalized[idx]
    
    print(f"Point order: lowest density → highest density")
    
    # =========================================================================
    # 5. CREATE FIGURE WITH CARTOPY
    # =========================================================================
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    # =========================================================================
    # 6. PLOT DENSITY SCATTER (SORTED)
    # =========================================================================
    scatter = ax.scatter(
        lon_sorted, lat_sorted,
        c=density_sorted,
        s=8,
        alpha=0.7,
        cmap='YlOrRd',
        norm=Normalize(vmin=0, vmax=1),
        transform=ccrs.PlateCarree(),
        edgecolors='none',
        rasterized=True  # For better PDF performance
    )
    
    # =========================================================================
    # 7. ADD GEOGRAPHIC FEATURES
    # =========================================================================
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
    
    # =========================================================================
    # 8. GRIDLINES
    # =========================================================================
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    
    # =========================================================================
    # 9. HORIZONTAL COLORBAR (HighResMIP style)
    # =========================================================================
    cbar = plt.colorbar(
        scatter,
        ax=ax,
        orientation='horizontal',
        shrink=0.6,
        aspect=30,
        pad=0.08
    )
    cbar.set_label(
        'Local Track Density (normalized)',
        fontsize=11,
        fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=10)
    
    # =========================================================================
    # 10. TITLE
    # =========================================================================
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    model = tdict['dataset']['model']
    exp = tdict['dataset']['exp']
    
    plt.title(
        f'TC Track Density Scatter Plot (sorted by density)\n'
        f'{startdate}–{enddate} | {model} {exp} | n={total_points:,} observations',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    
    # =========================================================================
    # 11. SAVE FIGURE
    # =========================================================================
    os.makedirs(tdict['paths']['plotdir'], exist_ok=True)
    
    # Clean names (remove spaces)
    model_clean = model.replace(" ", "_")
    exp_clean = exp.replace(" ", "_")
    startdate_clean = startdate.replace(" ", "").replace("-", "")
    enddate_clean = enddate.replace(" ", "").replace("-", "")
    
    save_path = os.path.join(
        tdict['paths']['plotdir'],
        f'density_scatter_{model_clean}_{exp_clean}_{startdate_clean}_{enddate_clean}.pdf'
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Density scatter plot saved to: {save_path}")
    
    plt.show()
    plt.close()






def plot_track_density_grid(trajfile, tdict, grid_size=2.5, max_timesteps=None):
    """
    Plot TC track density as transits per month using gridded counts.
    
    This function follows the HighResMIP-PRIMAVERA methodology (Page 10, Fig 3 OBS):
    - Divides the global domain into regular lat/lon grid cells
    - Counts TC passages through each cell
    - Normalizes by the time period to get "transits per month"
    - Uses DISCRETE logarithmic colorbar with distinct color bands
    - Custom colormap: white → brown → yellow → green → blue (HighResMIP style)
    
    Args:
        trajfile: Path to the filtered trajectory file
        tdict: Configuration dictionary with dataset info and paths
        grid_size: Grid cell size in degrees (default 2.5°)
        max_timesteps: Optional limit on trajectory length
    
    Returns:
        None (saves plot to file)
    """
    from datetime import datetime
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import os
    
    print(f"Creating track density grid from: {trajfile}")
    
    # =========================================================================
    # 1. READ TRAJECTORIES AND EXTRACT ALL POINTS
    # =========================================================================
    lon_all = []
    lat_all = []
    
    with open(trajfile, 'r') as f:
        for line in f:
            if 'track_id' in line or 'year' in line:
                continue
            
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            
            lon = float(parts[7])
            lat = float(parts[8])
            
            # Convert longitude from [0, 360] to [-180, 180]
            if lon > 180:
                lon -= 360
            
            lon_all.append(lon)
            lat_all.append(lat)
    
    lon_all = np.array(lon_all)
    lat_all = np.array(lat_all)
    
    total_points = len(lon_all)
    print(f"Total TC observation points: {total_points:,}")
    
    # =========================================================================
    # 2. CALCULATE TIME PERIOD IN MONTHS
    # =========================================================================
    startdate = tdict['time']['startdate']
    enddate = tdict['time']['enddate']
    
    # Parse dates (assuming format YYYYMMDD or YYYY-MM-DD)
    if '-' in startdate:
        start = datetime.strptime(startdate, '%Y-%m-%d')
        end = datetime.strptime(enddate, '%Y-%m-%d')
    else:
        start = datetime.strptime(startdate, '%Y%m%d')
        end = datetime.strptime(enddate, '%Y%m%d')
    
    # Calculate months
    n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    print(f"Time period: {n_months} months ({startdate} to {enddate})")
    
    # =========================================================================
    # 3. CREATE GRID AND COUNT TRANSITS
    # =========================================================================
    # Define grid
    lon_bins = np.arange(-180, 180 + grid_size, grid_size)
    lat_bins = np.arange(-50, 50 + grid_size, grid_size)
    
    # Create 2D histogram
    counts, lon_edges, lat_edges = np.histogram2d(
        lon_all, lat_all,
        bins=[lon_bins, lat_bins]
    )
    
    # Normalize by number of months
    transits_per_month = counts / n_months
    
    print(f"Grid resolution: {grid_size}° × {grid_size}°")
    print(f"Max transits per month: {transits_per_month.max():.2f}")
    if np.any(transits_per_month > 0):
        print(f"Min non-zero transits: {transits_per_month[transits_per_month > 0].min():.4f}")
    
    # =========================================================================
    # 4. DEFINE DISCRETE BOUNDARIES (LOGARITHMIC SPACING)
    # =========================================================================
    # Define discrete levels for transits per month
   # boundaries = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
   # n_levels = len(boundaries) - 1  # Number of discrete color bands
    
   # print(f"Using {n_levels} discrete color levels")
    
    # =========================================================================
    # 4. AUTOMATIC LOGARITHMIC BOUNDARIES
    # =========================================================================
    vmax = transits_per_month.max()
    vmin = transits_per_month[transits_per_month > 0].min() if np.any(transits_per_month > 0) else 0.01

    print(f"Data range: {vmin:.4f} to {vmax:.2f} transits/month")

   # Create 12 logarithmically-spaced boundaries
    n_levels = 12
    boundaries = np.logspace(np.log10(max(vmin, 0.01)), np.log10(max(vmax, 1.0)), n_levels + 1)
    boundaries[0] = 0.0  # Start from zero

    print(f"Automatic boundaries: {boundaries}")
    print(f"Using {n_levels} discrete color levels")
 
    # =========================================================================
    # 5. DISCRETIZE DATA INTO BANDS
    # =========================================================================
    # Assign each grid cell to a discrete level
    transits_discrete = np.digitize(transits_per_month, boundaries) - 1
    
    # Clip to valid range [0, n_levels-1]
    transits_discrete = np.clip(transits_discrete, 0, n_levels - 1)
    
    # Mask zeros (no transits = white/transparent)
    transits_discrete_masked = np.where(
        transits_per_month > 0,
        transits_discrete,
        np.nan
    )
    
    # =========================================================================
    # 6. CREATE DISCRETE COLORMAP
    # =========================================================================
    # Base colors: white → brown → yellow → green → blue
    base_colors = [
        '#FFFFFF',  # white
        '#8B4513',  # brown
        '#D2691E',  # chocolate
        '#FFD700',  # gold/yellow
        '#ADFF2F',  # green-yellow
        '#00FF00',  # green
        '#00CED1',  # turquoise
        '#0000FF'   # blue
    ]
    
    # Create continuous colormap first
    cmap_continuous = LinearSegmentedColormap.from_list('tc_density', base_colors, N=256)
    
    # Sample N discrete colors from the continuous colormap
    colors_discrete = [cmap_continuous(i / n_levels) for i in range(n_levels)]
    cmap_discrete = ListedColormap(colors_discrete)
    
    # =========================================================================
    # 7. CREATE FIGURE
    # =========================================================================
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    # =========================================================================
    # 8. PLOT GRIDDED DENSITY WITH DISCRETE COLORS
    # =========================================================================
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, transits_discrete_masked.T,
        cmap=cmap_discrete,
        vmin=0,
        vmax=n_levels,
        transform=ccrs.PlateCarree(),
        shading='auto'
    )
    
    # =========================================================================
    # 9. ADD GEOGRAPHIC FEATURES
    # =========================================================================
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=2, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    # =========================================================================
    # 10. GRIDLINES
    # =========================================================================
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
    # 11. HORIZONTAL COLORBAR (DISCRETE BANDS)
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
    cbar.set_label(
        'TC track density (transits per month)',
        fontsize=11,
        fontweight='bold'
    )
    
    # Set ticks at the CENTER of each color band
    tick_positions = np.arange(n_levels) + 0.5
    cbar.set_ticks(tick_positions)
    
    # Create labels showing the range for each band
    tick_labels = []
    for i in range(n_levels):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == n_levels - 1:  # Last band
            tick_labels.append(f'>{lower:.2g}')
        else:
            tick_labels.append(f'{lower:.2g}–{upper:.2g}')
    
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8, rotation=45)
    
    # =========================================================================
    # 12. TITLE
    # =========================================================================
    model = tdict['dataset']['model']
    exp = tdict['dataset']['exp']
    
    plt.title(
        f'TC Track Density (transits per month, discrete scale)\n'
        f'{startdate}–{enddate} | {model} {exp} | Grid: {grid_size}°',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    
    # =========================================================================
    # 13. SAVE FIGURE
    # =========================================================================
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







def plot_density_scatter_by_category(trajfile, tdict, max_timesteps=None, sample_size=10000):
    """
    Plot 6-panel density scatter: one subplot per Saffir-Simpson category.
    
    This function creates a comprehensive visualization showing spatial density
    patterns for each TC intensity category separately. The density is computed
    using Kernel Density Estimation (KDE) for each category independently.
    
    KDE METHODOLOGY:
    - For each category, extract all lon/lat points where TC reached that category
    - Compute 2D Gaussian KDE using scipy.stats.gaussian_kde
    - The KDE creates a smooth probability density field by placing a Gaussian
      "kernel" at each observation point and summing them
    - Bandwidth automatically selected by Scott's rule: optimal for normal data
    - Density normalized to [0-1] per category for consistent visualization
    - Higher values = more TC observations in that location for that category
    
    Args:
        trajfile: Path to the filtered trajectory file
        tdict: Configuration dictionary with dataset info and paths
        max_timesteps: Optional limit on trajectory length
        sample_size: Max points per category for KDE (default 10000)
    
    Returns:
        None (saves 6-panel plot to file)
    """
    print(f"Creating 6-panel density scatter by category from: {trajfile}")
    
    # =========================================================================
    # 1. READ TRAJECTORIES AND CLASSIFY BY CATEGORY
    # =========================================================================
    points_by_category = {cat: {'lon': [], 'lat': []} for cat in range(6)}
    
    with open(trajfile, 'r') as f:
        for line in f:
            if 'track_id' in line or 'year' in line:
                continue
            
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            
            lon = float(parts[7])
            lat = float(parts[8])
            slp = float(parts[9])  # Pa
            
            # Convert longitude
            if lon > 180:
                lon -= 360
            
            # Classify by category
            cat = category_from_slp_pa(slp)
            
            points_by_category[cat]['lon'].append(lon)
            points_by_category[cat]['lat'].append(lat)
    
    # Convert to numpy arrays
    for cat in range(6):
        points_by_category[cat]['lon'] = np.array(points_by_category[cat]['lon'])
        points_by_category[cat]['lat'] = np.array(points_by_category[cat]['lat'])
    
    # Print statistics
    print("\nPoints per category:")
    for cat in range(6):
        n_pts = len(points_by_category[cat]['lon'])
        print(f"  Cat {cat}: {n_pts:,} points")
    
    # =========================================================================
    # 2. CREATE 6-PANEL FIGURE
    # =========================================================================
    fig = plt.figure(figsize=(18, 12))
    
    cat_names = [
        'TD (≥1005 hPa)',
        'Cat 1 (990–1004 hPa)',
        'Cat 2 (975–989 hPa)',
        'Cat 3 (960–974 hPa)',
        'Cat 4 (945–959 hPa)',
        'Cat 5 (<945 hPa)'
    ]
    
    cmaps = ['Greens', 'YlGn', 'YlOrBr', 'Oranges', 'OrRd', 'RdPu']
    
    # =========================================================================
    # 3. PLOT EACH CATEGORY
    # =========================================================================
    for cat in range(6):
        ax = plt.subplot(2, 3, cat + 1, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
        
        lon_cat = points_by_category[cat]['lon']
        lat_cat = points_by_category[cat]['lat']
        n_points = len(lon_cat)
        
        if n_points == 0:
            # Empty category - just show base map
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.set_title(f'{cat_names[cat]}\n(no observations)', 
                        fontsize=11, fontweight='bold')
            continue
        
        # Sample if too many points
        if n_points > sample_size:
            indices = np.random.choice(n_points, sample_size, replace=False)
            lon_sample = lon_cat[indices]
            lat_sample = lat_cat[indices]
            print(f"  Cat {cat}: Sampled {sample_size:,} from {n_points:,} points")
        else:
            lon_sample = lon_cat
            lat_sample = lat_cat
        
        # Compute KDE
        try:
            xy = np.vstack([lon_sample, lat_sample])
            kde = gaussian_kde(xy)
            density = kde(xy)
            density_normalized = density / density.max()
            
            # Plot scatter with density coloring
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
            
            # HORIZONTAL COLORBAR below each subplot
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
            print(f"  Cat {cat}: KDE failed ({e}), using simple scatter")
            ax.scatter(
                lon_sample, lat_sample,
                s=3, alpha=0.5, color='blue',
                transform=ccrs.PlateCarree()
            )
        
        
        # Geographic features
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0, alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        
        # Gridlines
        gl = ax.gridlines(
            draw_labels=True if cat >= 3 else False,  # Labels only bottom row
            linewidth=0.3,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        
        # Title
        ax.set_title(
            f'{cat_names[cat]}\n(n={n_points:,} obs)',
            fontsize=11,
            fontweight='bold'
        )
    
    # =========================================================================
    # 4. OVERALL TITLE
    # =========================================================================
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
    
    # =========================================================================
    # 5. SAVE FIGURE
    # =========================================================================
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
    print(f"\n6-panel density scatter saved to: {save_path}")
    
    plt.show()
    plt.close()