# tc_data_manager.py
#
# TROPICAL CYCLONE DATA MANAGER - IN-MEMORY STORAGE
# ==================================================
#
# # Version: LIST numerical ID and sequential
# This module provides a class to load TC trajectory data once and keep it
# in memory for fast repeated access, avoiding slow disk I/O operations.



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
        trajectories: LIST of storm dictionaries (ordered)
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
        self.trajectories = None  # Now a LIST!
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
        
        # Convert to numpy arrays
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
        
        # Build trajectory LIST (not dict!)
        print(f"  Building trajectory list...")
        self._build_trajectories()
    
    def _build_trajectories(self):
        """Organize data by individual storm trajectories - AS A LIST."""
        self.trajectories = []  # LISTA invece di dizionario!
        
        # Get unique storm IDs and sort them numerically
        unique_ids = np.unique(self.data['storm_id'])
        
        # Sort numerically (importante per avere ordine corretto!)
        try:
            unique_ids_sorted = sorted(unique_ids, key=lambda x: int(x))
        except ValueError:
            # Se non sono tutti numerici, ordina come stringhe
            unique_ids_sorted = sorted(unique_ids)
        
        for storm_id in unique_ids_sorted:
            mask = self.data['storm_id'] == storm_id
            
            storm_dict = {
                'id': storm_id,  # Memorizziamo l'ID dentro il dict
                'lon': self.data['lon'][mask],
                'lat': self.data['lat'][mask],
                'slp': self.data['slp'][mask],
                'wind': self.data['wind'][mask],
                'year': self.data['year'][mask],
                'month': self.data['month'][mask],
                'day': self.data['day'][mask],
                'hour': self.data['hour'][mask]
            }
            
            self.trajectories.append(storm_dict)
    
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
        Get trajectory list.
        
        Returns:
            list: List of storm dictionaries, each with:
                  {'id': str, 'lon': array, 'lat': array, 'slp': array, ...}
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
        
        print(f"\nTrajectory structure:")
        print(f"  Type: LIST (not dict)")
        print(f"  Length: {len(self.trajectories)}")
        if len(self.trajectories) > 0:
            print(f"  First storm ID: '{self.trajectories[0]['id']}'")
            print(f"  Last storm ID: '{self.trajectories[-1]['id']}'")
        
        print(f"{'='*70}\n")
    
    def __repr__(self):
        """String representation of the manager."""
        return f"TCDataManager({self.n_obs:,} obs, {self.n_storms} storms, LIST structure)"


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