# ===========================================
# MODIFIED PLOTTING FUNCTIONS FOR DIRECT FORMAT
# ===========================================
# plotting_TCs_custom.py

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import os
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable



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
            s=15,
            linewidths=0.5,
            marker=".",
            alpha=0.8,
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

    save_path = os.path.join(
      tdict['paths']['plotdir'],
      f"tracks_colored_{color_by}_{tdict['dataset']['model']}_{tdict['dataset']['exp']}_{startdate}_{enddate}.pdf"
    )

    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    
    print(f"Plot saved to: {save_path}")
    plt.show()
