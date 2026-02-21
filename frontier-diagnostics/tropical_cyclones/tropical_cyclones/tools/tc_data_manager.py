# tc_data_manager.py
#
# TROPICAL CYCLONE DATA MANAGER - IN-MEMORY STORAGE
# ==================================================
#
# This module provides a class to load TC trajectory data once and keep it
# in memory for fast repeated access, avoiding slow disk I/O operations.
#
# PERFORMANCE BENEFITS:
# ---------------------
# - Reading from file: ~100-1000 ms per operation (disk I/O)
# - Reading from memory: ~0.1-1 ms per operation (RAM access)
# - Speed improvement: 100-10,000x faster!
#
# WHY IT'S FASTER:
# ----------------
# 1. DISK I/O is SLOW:
#    - Physical disk read/write operations take milliseconds
#    - File system overhead (opening, seeking, buffering)
#    - Operating system caching may not always help
#
# 2. RAM ACCESS is FAST:
#    - Direct memory access takes nanoseconds
#    - No file system overhead
#    - Data is immediately available
#
# 3. ONE-TIME COST:
#    - Load data ONCE from disk → pay the I/O cost once
#    - Access data MANY TIMES from RAM → instant access
#
# USAGE:
# ------
# # Load data once (cella 5 nel notebook)
# from tc_data_manager import TCDataManager
# tc_data = TCDataManager(filtered_file)
# tc_data.summary()
#
# # Use multiple times - instant! (cella 6, 7, 8...)
# plot_trajectories_direct(tc_data, config)
# plot_density_scatter(tc_data, config)
# plot_track_density_grid(tc_data, config, grid_size=1.5)
#
# HOW IT WORKS:
# -------------
# The class loads the entire trajectory file into structured numpy arrays:
#   - storm_id, year, month, day, hour
#   - lon, lat, slp, wind, zs
#
# All data is stored in RAM as numpy arrays (very fast access).
# Additional structures:
#   - trajectories: dictionary organized by storm_id
#   - metadata: time period, statistics, etc.
#
# When a plotting function needs data, it retrieves it directly from RAM
# instead of reading from disk → 1000x faster!
#
# ===============================================================================


import numpy as np
from datetime import datetime


class TCDataManager:
    """
    Manages TC trajectory data in memory for fast repeated access.
    
    Loads trajectory data once from file and provides fast in-memory access
    to all plotting and analysis functions.
    
    Attributes:
        trajfile: Original file path
        data: Dictionary of numpy arrays (all observations)
        trajectories: Dictionary organized by storm
        n_obs: Total number of observations
        n_storms: Total number of unique storms
    """
    
    def __init__(self, trajfile):
        """
        Load TC trajectory data into memory.
        
        Args:
            trajfile: Path to trajectory file (filtered or unfiltered)
        """
        self.trajfile = trajfile
        self.data = None
        self.trajectories = None
        self.metadata = {}
        
        print(f"\n{'='*70}")
        print(f"LOADING TC DATA INTO MEMORY")
        print(f"{'='*70}")
        print(f"Source: {trajfile}")
        
        self._load_data()
        
        print(f"✓ Data loaded successfully!")
        print(f"  Observations: {self.n_obs:,}")
        print(f"  Storms: {self.n_storms}")
        print(f"{'='*70}\n")
    
    def _load_data(self):
        """Load all data from file into structured arrays."""
        # Lists to store all data
        storm_ids = []
        years = []
        months = []
        days = []
        hours = []
        i_coords = []
        j_coords = []
        lons = []
        lats = []
        slps = []
        winds = []
        zs_values = []
        
        line_count = 0
        with open(self.trajfile, 'r') as f:
            for line_num, line in enumerate(f):
                # Skip header
                if line_num == 0 or 'track_id' in line or 'year' in line:
                    continue
                
                parts = line.strip().split()
                if len(parts) != 12:
                    continue
                
                try:
                    storm_ids.append(parts[0])
                    years.append(int(parts[1]))
                    months.append(int(parts[2]))
                    days.append(int(parts[3]))
                    hours.append(int(parts[4]))
                    i_coords.append(int(parts[5]))
                    j_coords.append(int(parts[6]))
                    lons.append(float(parts[7]))
                    lats.append(float(parts[8]))
                    slps.append(float(parts[9]))
                    winds.append(float(parts[10]))
                    zs_values.append(float(parts[11]))
                    line_count += 1
                except (ValueError, IndexError):
                    continue
        
        print(f"  Parsed {line_count:,} observations from file")
        
        # Convert to numpy arrays (even faster access!)
        print(f"  Converting to numpy arrays...")
        self.data = {
            'storm_id': np.array(storm_ids),
            'year': np.array(years),
            'month': np.array(months),
            'day': np.array(days),
            'hour': np.array(hours),
            'i': np.array(i_coords),
            'j': np.array(j_coords),
            'lon': np.array(lons),
            'lat': np.array(lats),
            'slp': np.array(slps),
            'wind': np.array(winds),
            'zs': np.array(zs_values)
        }
        
        # Store metadata
        self.n_obs = len(self.data['storm_id'])
        self.n_storms = len(np.unique(self.data['storm_id']))
        
        # Build trajectory dictionary (organized by storm)
        print(f"  Building trajectory dictionary...")
        self._build_trajectories()
    
    def _build_trajectories(self):
        """Organize data by individual storm trajectories."""
        self.trajectories = {}
        
        for storm_id in np.unique(self.data['storm_id']):
            mask = self.data['storm_id'] == storm_id
            
            self.trajectories[storm_id] = {
                'lon': self.data['lon'][mask],
                'lat': self.data['lat'][mask],
                'slp': self.data['slp'][mask],
                'wind': self.data['wind'][mask],
                'year': self.data['year'][mask],
                'month': self.data['month'][mask],
                'day': self.data['day'][mask],
                'hour': self.data['hour'][mask]
            }
    
    def get_all_points(self):
        """
        Get all TC observation points as lon/lat arrays.
        
        Returns:
            lon, lat: numpy arrays of all observations
        """
        return self.data['lon'].copy(), self.data['lat'].copy()
    
    def get_all_data(self):
        """
        Get complete data dictionary.
        
        Returns:
            dict: Full data dictionary with all fields
        """
        return self.data
    
    def get_trajectories(self):
        """
        Get trajectory dictionary organized by storm.
        
        Returns:
            dict: {storm_id: {'lon': array, 'lat': array, 'slp': array, ...}}
        """
        return self.trajectories
    
    def get_time_period(self):
        """
        Get time period covered by the data.
        
        Returns:
            start_date, end_date, n_months
        """
        years = self.data['year']
        months = self.data['month']
        
        start = datetime(years.min(), months[years == years.min()].min(), 1)
        end = datetime(years.max(), months[years == years.max()].max(), 1)
        
        n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        
        return start, end, n_months
    
    def summary(self):
        """Print summary statistics."""
        print(f"\n{'='*70}")
        print(f"TC DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Source file: {self.trajfile}")
        print(f"Total observations: {self.n_obs:,}")
        print(f"Total storms: {self.n_storms}")
        
        start, end, n_months = self.get_time_period()
        print(f"\nTime period:")
        print(f"  Start: {start.strftime('%Y-%m')}")
        print(f"  End: {end.strftime('%Y-%m')}")
        print(f"  Duration: {n_months} months")
        
        print(f"\nData ranges:")
        print(f"  SLP: {self.data['slp'].min():.1f} - {self.data['slp'].max():.1f} Pa")
        print(f"  Wind: {self.data['wind'].min():.1f} - {self.data['wind'].max():.1f} m/s")
        print(f"  Latitude: {self.data['lat'].min():.1f}° to {self.data['lat'].max():.1f}°")
        print(f"  Longitude: {self.data['lon'].min():.1f}° to {self.data['lon'].max():.1f}°")
        
        print(f"\nMemory usage:")
        total_mb = sum(arr.nbytes for arr in self.data.values()) / 1024 / 1024
        print(f"  Total: {total_mb:.2f} MB")
        print(f"{'='*70}\n")
    
    def __repr__(self):
        """String representation of the manager."""
        return f"TCDataManager({self.n_obs:,} obs, {self.n_storms} storms)"


# ===============================================================================
# HELPER FUNCTIONS FOR PLOTTING COMPATIBILITY
# ===============================================================================

def is_tc_data_manager(data):
    """Check if input is a TCDataManager object."""
    return isinstance(data, TCDataManager)


def ensure_tc_data(data_or_file):
    """
    Convert file path to TCDataManager if needed.
    
    Args:
        data_or_file: Either TCDataManager object or file path string
    
    Returns:
        TCDataManager object
    """
    if is_tc_data_manager(data_or_file):
        return data_or_file
    else:
        # Load from file
        return TCDataManager(data_or_file)


# ===============================================================================
# USAGE EXAMPLE
# ===============================================================================
#if __name__ == "__main__":
#    print("TC Data Manager - Example Usage")
#    print("="*70)
    
    # Example 1: Load data
#    filtered_file = "path/to/filtered_trajectories.txt"
#    tc_data = TCDataManager(filtered_file)
    
    # Example 2: Print summary
#    tc_data.summary()
    
    # Example 3: Access data
#    lon, lat = tc_data.get_all_points()
#    print(f"Got {len(lon):,} points from memory (instant!)")
    
#    trajectories = tc_data.get_trajectories()
#    print(f"Got {len(trajectories)} storm trajectories from memory (instant!)")
    
    # Example 4: Use with plotting functions
    # (These would be in your plotting_TCs_custom_memory.py)
    # plot_trajectories_direct(tc_data, config)
    # plot_density_scatter(tc_data, config)