#!/usr/bin/env python3
"""
set_regions.py
==============

This script updates element regions based on coordinate boundaries extracted from a CSV file.
It:
  - Reads 'cobi.csv' and 'heart.elem' from the openCarp folder of the specified mesh.
  - Uses UVC coordinate bounds (converted to CoBi) to filter CSV rows.
  - Writes a mask (.dat) file indicating matching rows.
  - Updates the element file (.elem) using the matching rows.

Modules from the "csv_elem_functions" folder and "points_finder" are used.
"""

import os
import sys
import csv
import math
import csv
import time
import os
import tempfile
from helper import uvc_to_cobi        

# --------------------------------------------------------------------------
# 1) Initial path setup
# --------------------------------------------------------------------------
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Point to the "csv_elem_functions" folder inside "Modules"
csv_elem_dir = os.path.join(parent_dir, "Modules", "csv_elem_functions")
# Add the new path to the system path
sys.path.insert(0, csv_elem_dir)

# --------------------------------------------------------------------------
# 2) Import modules from "csv_elem_functions" and "points_finder"
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# 3) Helper function to create a mask .dat file
# --------------------------------------------------------------------------
def write_mask_dat_for_csv(csv_path, matching_rows, dat_path="mask.dat"):
    """
    Reads the CSV at `csv_path` and writes a .dat file with one line per row.
    Each line contains '1' if the row index is in matching_rows, or '0' otherwise.

    Args:
        csv_path (str): Path to the CSV (e.g. "cobi.csv").
        matching_rows (list or set of int): 0-based row indices to flag as '1'.
        dat_path (str): Path for the output file (default: "mask.dat").

    Returns:
        str: The path to the output .dat file.
    """
    matching_rows_set = set(matching_rows)

    with open(csv_path, 'r') as f:
        csv_lines = f.readlines()
    num_rows = len(csv_lines)

    with open(dat_path, 'w') as dat_file:
        for i in range(num_rows):
            val = "1" if i in matching_rows_set else "0"
            dat_file.write(val + "\n")

    return dat_path

# --------------------------------------------------------------------------
# 4) Main regions pipeline as a callable function
# --------------------------------------------------------------------------
def run_set_regions(folder: str = "Mean") -> None:
    """
    Updates element regions based on coordinate boundaries.

    Parameters:
      folder (str): Name of the folder where mesh data is stored (e.g., "Mean").
    
    The script expects input files 'cobi.csv' and 'heart.elem' in:
         <Meshes>/<folder>/openCarp/
    and writes the mask file and updated element file in that folder.
    """
    # Construct the target folder (where the input files are located)
    target_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "Meshes", folder, "openCarp")
    )
    
    # Define input file names and paths
    csv_file = os.path.join(target_folder, "cobi.csv")
    elem_file = os.path.join(target_folder, "heart.elem")
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        sys.exit(1)
    if not os.path.exists(elem_file):
        print(f"Element file not found: {elem_file}")
        sys.exit(1)
    
    # Define UVC bounds for filtering.
    # These example values can be adjusted as needed.
    a_min, a_max = 0.15, 0.9
    r_min, r_max = 0, 0.0  # Placeholder, not used
    m_min, m_max = 0.0, 0.3
    v_val = 0            # Placeholder, not used

    # Convert lower UVC to CoBi
    ab_lower, rt_lower, tm_lower, tv_lower = uvc_to_cobi(a_min, r_min, m_min, v_val)
    # Convert upper UVC to CoBi
    ab_upper, rt_upper, tm_upper, tv_upper = uvc_to_cobi(a_max, r_max, m_max, v_val)

    ab_bounds = (min(ab_lower, ab_upper), max(ab_lower, ab_upper))
    rt_bounds = (0, 1)
    tm_bounds = (min(tm_lower, tm_upper), max(tm_lower, tm_upper))
    tv_bounds = (0, 1)

    print("tm bounds:", tm_bounds)
    print("ab bounds:", ab_bounds)

    # Get matching CSV row indices based on the specified boundaries.
    matching_rows = get_rows_within_coordinates(csv_file, ab_bounds, rt_bounds, tm_bounds, tv_bounds)
    print(f"Found {len(matching_rows)} matching rows based on the coordinate bounds.")

    # Write the mask file (mask.dat) in the target folder.
    mask_dat_path = os.path.join(target_folder, "mask.dat")
    write_mask_dat_for_csv(csv_file, matching_rows, mask_dat_path)
    print(f"Mask file written to: {mask_dat_path}")

    # Define the new region id to set (for example, region 1)
    region_id = 1
    updated_elem_path = update_elem_file_quick(elem_file, matching_rows, region_id)
    print("Updated element file written to:", updated_elem_path)

# Allow calling this script directly.
if __name__ == "__main__":
    run_set_regions()


def get_rows_within_coordinates(csv_filename, ab_bounds, rt_bounds, tm_bounds, tv_bounds):
    """
    Reads the CSV file and returns a list of row indices (starting from 0 for the first data row)
    for which the values in columns 'ab', 'rt', 'tm', and 'tv' are within the provided bounds.

    Parameters:
      csv_filename (str): Path to the CSV file.
      ab_bounds (tuple): (lower_bound, upper_bound) for 'ab' column.
      rt_bounds (tuple): (lower_bound, upper_bound) for 'rt' column.
      tm_bounds (tuple): (lower_bound, upper_bound) for 'tm' column.
      tv_bounds (tuple): (lower_bound, upper_bound) for 'tv' column.

    Returns:
      list: List of integer row indices that satisfy all boundary conditions.
    """
    start_time=time.time()
    matching_rows = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            try:
                ab = float(row['ab'])
                rt = float(row['rt'])
                tm = float(row['tm'])
                tv = float(row['tv'])
            except ValueError:
                # skip rows that cannot be converted
                continue

            if (ab_bounds[0] <= ab <= ab_bounds[1] and
                rt_bounds[0] <= rt <= rt_bounds[1] and
                tm_bounds[0] <= tm <= tm_bounds[1] and
                tv_bounds[0] <= tv <= tv_bounds[1]):
                matching_rows.append(idx)
                end_time = time.time()  # End timing
    print(f"get_rows_within_coordinates took {end_time - start_time:.6f} seconds")
    return matching_rows


def update_elem_file_quick(elem_filename, row_indices, region_id, output_filename=None, reset_all=False):
    """
    Reads an element (.elem) file and updates the region id (last field) for the elements.
    
    If no output_filename is given, the file is updated "in place" by writing to a
    temporary file and then replacing the original file with the temporary file.
    
    Parameters:
      elem_filename (str): Path to the .elem file.
      row_indices (list or set): Coordinate values (as integers) to check in each row.
      region_id (int or str): The region id value to set for the selected elements.
      output_filename (str, optional): File to write updated content. If None, does in-place update.
      reset_all (bool): If True, resets all region ids to 0. Default is False.

    Returns:
      str: The name of the file where the updated content is written.
    """
    start_time = time.time()
    
    # Convert row_indices to a set for O(1) membership checks
    if not isinstance(row_indices, set):
        row_indices = set(row_indices)
    
    # Decide if we're doing an in-place update (using a temp file) or writing to a separate file
    in_place = (output_filename is None)
    if in_place:
        # Create a temporary file to write the output
        temp_file = tempfile.NamedTemporaryFile('w', delete=False)
        output_filename = temp_file.name
        temp_file.close()  # We'll reopen it below
    
    with open(elem_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        # 1) Read/write the header line
        header_line = next(infile, None)
        if header_line is None:
            print("Empty element file.")
            return elem_filename if in_place else output_filename
        outfile.write(header_line)
        
        # 2) Process each subsequent line
        for line in infile:
            parts = line.strip().split()
            if not parts:
                outfile.write("\n")
                continue
            
            if reset_all:
                # If reset_all is True, set the region id (last token) to "0"
                parts[-1] = "0"
            else:
                # Check if any coordinate in [1:-1] is in row_indices
                coord_tokens = parts[1:-1]
                if any(int(token) in row_indices for token in coord_tokens):
                    parts[-1] = str(region_id)
            
            outfile.write(" ".join(parts) + "\n")
    
    if in_place:
        # We wrote to a temp file; now replace the original file
        os.replace(output_filename, elem_filename)
        output_filename = elem_filename
    
    end_time = time.time()
    print(f"update_elem_file took {end_time - start_time:.6f} seconds")

    return output_filename


