import os
import sys
import math
import csv
from helper import uvc_to_cobi       


# --------------------------------------------------------------------------
# 3) Helper functions
# --------------------------------------------------------------------------
def create_vtx_file(vtx_path: str, matched_indices: list[int]) -> None:
    """
    Creates a .vtx file listing the matched indices.
    Each index is written on its own line.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(vtx_path), exist_ok=True)
    with open(vtx_path, "w") as f:
        f.write(f"{len(matched_indices)}\n")
        f.write("extra\n")  # Can be adjusted if needed
        for idx in matched_indices:
            f.write(f"{idx}\n")


def combine_lists_and_write_dat(
    csv_file: str,
    list_of_index_lists: list[list[int]],
    output_dat: str = "combined.dat",
):
    """
    Reads the CSV file to determine the number of data rows (excluding the header),
    creates an array of 0/1 flags, and writes the flags to a .dat file.
    """
    # 1) Determine the number of data rows in the CSV (ignoring the header)
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        row_count = len(rows)

    # 2) Initialize an array of zeros
    flags = [0] * row_count

    # 3) For each index in any sub-list, mark it as 1
    for indices in list_of_index_lists:
        for idx in indices:
            if 0 <= idx < row_count:
                flags[idx] = 1

    # 4) Write the flags to the .dat file, one per line
    with open(output_dat, "w") as out:
        for flag in flags:
            out.write(f"{flag}\n")

    print(f"Created {output_dat} with {row_count} lines (0 or 1).")


# --------------------------------------------------------------------------
# 4) Main electrodes pipeline as a callable function
# --------------------------------------------------------------------------
def run_electrodes(folder: str = "Mean", instance="original") -> None:
    """
    Runs the electrode placement pipeline.

    Parameters:
      folder (str): Name of the folder where the mesh data is stored (e.g., "Mean").

    The script expects a 'cobi.csv' file in:
         ../../Meshes/<folder>/openCarp/cobi.csv
    and writes output files (including .dat and .vtx files) into the same openCarp folder,
    with electrode-specific outputs placed in a subfolder 'Electrodes'.
    """
    # Construct the target folder (where the openCarp data is located)
    target_folder = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "Meshes",
            folder,
            "openCarp",
        )
    )

    # CSV file path (CoBi data)
    csv_file = os.path.join(target_folder, "cobi.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        sys.exit(1)

    # Global parameters for all circles
    thickness = 0.05
    radius = 0.05

    # Five UVC base coordinate definitions:
    # Each tuple: (circle name, a, r, m, v)
    circles_uvc = [
        ("lvsf", 0.61, 0.73, 0.0, 0),
        ("lvpf", 0.47, -1.36, 0.0, 0),
        ("lvaf", 0.82, 1.94, 0.0, 0),
        ("rvsf", 0.73, -0.04, 0.0, 0),
        ("rvmod", 0.63, 0.21, 0.0, 1),
    ]

    # List to store matched indices from all circles
    all_indices = []

    # Ensure the Electrodes subfolder exists
    electrodes_folder = os.path.join(target_folder, "Electrodes")
    os.makedirs(electrodes_folder, exist_ok=True)

    # Process each circle
    for circle_name, a_val, r_val, m_val, v_val in circles_uvc:
        print(f"\n=== Processing {circle_name} ===")
        # Convert UVC coordinates to CoBi coordinates
        ab_base, rt_base, tm_base, tv_base = uvc_to_cobi(a_val, r_val, m_val, v_val)
        print(
            f"  UVC -> CoBi: (a={a_val}, r={r_val}, m={m_val}, v={v_val}) -> "
            f"(ab={ab_base:.3f}, rt={rt_base:.3f}, tm={tm_base:.3f}, tv={tv_base:.3f})"
        )
        if circle_name == "rvsf":
            tv_base = 1  # Adjusted conversion for split ventricular

        # Build path for the .dat file for this circle
        dat_file = os.path.join(target_folder, f"{circle_name}.dat")
        # Find points in a cylinder and create the .dat file
        matched_indices = find_points_in_cylinder(
            csv_file=csv_file,
            base_ab=ab_base,
            base_rt=rt_base,
            base_tm=tm_base,
            base_tv=tv_base,
            radius=radius,
            thickness=thickness,
            dat_file=dat_file,
        )
        print(f"  Matched {len(matched_indices)} rows. .dat -> {dat_file}")

        # Create a .vtx file for this circle
        vtx_file = os.path.join(electrodes_folder, f"{circle_name}.vtx")
        create_vtx_file(vtx_file, matched_indices)
        print(f"  .vtx file created -> {vtx_file}")

        all_indices.append(matched_indices)

    print("\nAll circles processed successfully.")

    # Combine all matched indices into one .dat file
    combined_dat = os.path.join(electrodes_folder, "combined_circles.dat")
    combine_lists_and_write_dat(csv_file, all_indices, output_dat=combined_dat)


def find_points_in_cylinder(
    csv_file: str,
    base_ab: float,
    base_rt: float,
    base_tm: float,
    base_tv: float,
    radius: float,
    thickness: float,
    dat_file: str = "output.dat",
) -> list[int]:
    """
    Reads a CSV file with header [ab, rt, tm, tv] and finds all rows whose
    coordinates lie within a 'cylindrical' region around the given base coordinate.

    Conditions:
      1) |tm - base_tm| < thickness
      2) (ab - base_ab)^2 + (rt - base_rt)^2 < radius^2
      3) tv == base_tv

    Creates a .dat file (one line per CSV row) where each line is 1 if that row
    meets the conditions, else 0.

    :param csv_file:   Path to the CSV file.
    :param base_ab:    ab-value of the base coordinate.
    :param base_rt:    rt-value of the base coordinate.
    :param base_tm:    tm-value of the base coordinate.
    :param base_tv:    tv-value of the base coordinate.
    :param radius:     Radius for the cylinder in the ab-rt plane.
    :param thickness:  Max allowed difference in tm from base_tm.
    :param dat_file:   Path to the output .dat file.
    :return:           A list of row indices (0-based) that match all conditions.
    """
    matched_indices = []
    acceptance_flags = []  # Will hold 1 or 0 for each row in the CSV

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        row_index = 0

        for row in reader:
            # Parse floats from the CSV columns
            ab_val = float(row["ab"])
            rt_val = float(row["rt"])
            tm_val = float(row["tm"])
            tv_val = float(row["tv"])

            # Check conditions:
            # 1) thickness in tm
            if tv_val == base_tv:
                # 2) radial distance in ab-rt plane
                dist_sq1 = (ab_val - base_ab) ** 2 + (rt_val - base_rt) ** 2
                dist_sq2 = (ab_val - base_ab) ** 2 + (rt_val - base_rt - 1) ** 2
                dist_sq3 = (ab_val - base_ab) ** 2 + (rt_val - base_rt + 1) ** 2
                if dist_sq1 < radius**2 or dist_sq2 < radius**2 or dist_sq3 < radius**2:
                    # 3) same tv
                    if (
                        abs(tm_val - base_tm) < thickness
                    ):  
                        # If all conditions pass:

                        matched_indices.append(row_index)
                        row_index += 1
                        continue

            # If we reach here, at least one condition failed

            row_index += 1

    return matched_indices



