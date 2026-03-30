"""Functions for the analysis of tempest-extrems output (.txt files) from https://github.com/zarzycki/cymep"""

import re

import numpy as np


def get_trajectories(filename, n_vars, header_delim_str, is_unstruc):
    """
    Retrieves trajectories from a TempestExtremes file (.txt file generated from StitchNodes).

    Args:
        filename: Name of the TempestExtremes file.
        n_vars: Number of variables in each data point. Set to -1 to automatically determine from the data.
        header_delim_str: Delimiter string to identify header lines.
        is_unstruc: Boolean flag indicating whether the data is unstructured.

    Returns:
        numtraj: Total number of trajectories.
        max_num_pts: Maximum length of any trajectory.
        prodata: Numpy array containing the trajectory data.
    """

    print("Getting trajectories from TempestExtremes file...")
    print("Running get_trajectories on '%s' with unstruc set to '%s'" %
          (filename, is_unstruc))
    print("n_vars set to %d and header_delim_str set to '%s'" %
          (n_vars, header_delim_str))

    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()

    # Find total number of trajectories and maximum length of trajectories
    numtraj = 0
    num_pts = []
    for line in data:
        if header_delim_str in line:
            # if header line, store number of points in given traj in num_pts
            head_arr = line.split()
            numtraj += 1
            num_pts.append(int(head_arr[1]))
        else:
            # if not a header line, and n_vars = -1, find number of columns in data point
            if n_vars < 0:
                n_vars = len(line.split())

    max_num_pts = max(num_pts)  # Maximum length of ANY trajectory

    print("Found %d columns" % n_vars)
    print("Found %d trajectories" % numtraj)

    # Initialize storm and line counter
    storm_id = -1
    line_of_traj = -1

    # Create array for data
    if is_unstruc:
        prodata = np.empty((n_vars+1, numtraj, max_num_pts))
    else:
        prodata = np.empty((n_vars, numtraj, max_num_pts))

    prodata[:] = np.nan

    for _, line in enumerate(data):
        if header_delim_str in line:  # check if header string is satisfied
            storm_id += 1      # increment storm
            line_of_traj = 0    # reset trajectory line to zero
        else:
            pt_arr = line.split()
            for jj in range(n_vars):
                if is_unstruc:
                    prodata[jj+1, storm_id, line_of_traj] = pt_arr[jj]
                else:
                    prodata[jj, storm_id, line_of_traj] = pt_arr[jj]
            line_of_traj += 1   # increment line

    print("... done reading data")
    return numtraj, max_num_pts, prodata


def get_nodes(filename, n_vars, is_unstruc):
    """
    Retrieves nodes from a TempestExtremes file (.txt output from DetectNodes).

    Args:
        filename: Name of the TempestExtremes file.
        n_vars: Number of variables in each data point. Set to -1 to automatically determine from the data.
        is_unstruc: Boolean flag indicating whether the data is unstructured.

    Returns:
        numnodetimes: Total number of nodes.
        max_num_pts: Maximum length of any node trajectory.
        prodata: Numpy array containing the node data.
    """
    print("Getting nodes from TempestExtremes file...")

    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()

    # Find total number of trajectories and maximum length of trajectories
    numnodetimes = 0
    num_pts = []
    for line in data:
        if re.match(r'\w', line):
            # if header line, store number of points in given traj in num_pts
            head_arr = line.split()
            numnodetimes += 1
            num_pts.append(int(head_arr[3]))
        else:
            # if not a header line, and n_vars = -1, find number of columns in data point
            if n_vars < 0:
                n_vars = len(line.split())

        max_num_pts = max(num_pts)  # Maximum length of ANY trajectory

    print("Found %d columns" % n_vars)
    print("Found %d trajectories" % numnodetimes)
    print("Found %d max_num_pts" % max_num_pts)

    # Initialize storm and line counter
    storm_id = -1
    line_of_traj = -1

    # Create array for data
    if is_unstruc:
        prodata = np.empty((n_vars+5, numnodetimes, max_num_pts))
    else:
        prodata = np.empty((n_vars+4, numnodetimes, max_num_pts))

    prodata[:] = np.NAN

    # nextHeadLine = 0
    for _, line in enumerate(data):
        if re.match(r'\w', line):  # check if header string is satisfied
            storm_id += 1      # increment storm
            line_of_traj = 0    # reset trajectory line to zero
            head_arr = line.split()
            yyyy = int(head_arr[0])
            mm = int(head_arr[1])
            dd = int(head_arr[2])
            hh = int(head_arr[4])
        else:
            pt_arr = line.split()
            for jj in range(n_vars-1):
                if is_unstruc:
                    prodata[jj+1, storm_id, line_of_traj] = pt_arr[jj]
                else:
                    prodata[jj, storm_id, line_of_traj] = pt_arr[jj]
            if is_unstruc:
                prodata[n_vars+1, storm_id, line_of_traj] = yyyy
                prodata[n_vars+2, storm_id, line_of_traj] = mm
                prodata[n_vars+3, storm_id, line_of_traj] = dd
                prodata[n_vars+4, storm_id, line_of_traj] = hh
            else:
                prodata[n_vars, storm_id, line_of_traj] = yyyy
                prodata[n_vars+1, storm_id, line_of_traj] = mm
                prodata[n_vars+2, storm_id, line_of_traj] = dd
                prodata[n_vars+3, storm_id, line_of_traj] = hh
            line_of_traj += 1   # increment line

    print("... done reading data")
    return numnodetimes, max_num_pts, prodata
