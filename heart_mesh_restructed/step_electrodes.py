import os, sys, csv, math
from math_utils import uvc_to_cobi

# --- helpers (verbatim) -----------------------------------------------------
def create_vtx_file(vtx_path: str, matched_indices: list[int]) -> None:
    os.makedirs(os.path.dirname(vtx_path), exist_ok=True)
    with open(vtx_path, "w") as f:
        f.write(f"{len(matched_indices)}\nextra\n")
        for idx in matched_indices:
            f.write(f"{idx}\n")

def combine_lists_and_write_dat(csv_file, list_of_index_lists, output_dat="combined.dat"):
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    flags = [0]*len(rows)
    for indices in list_of_index_lists:
        for idx in indices:
            if 0 <= idx < len(flags):
                flags[idx] = 1
    with open(output_dat, "w") as out:
        for flag in flags:
            out.write(f"{flag}\n")

def run_electrodes(folder: str = "Mean", instance="original") -> None:
    target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "Meshes", folder, "openCarp"))
    csv_file      = os.path.join(target_folder, "cobi.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}"); sys.exit(1)

    thickness = 0.05; radius = 0.05
    circles_uvc = [
        ("lvsf", 0.61,  0.73, 0.0, 0),
        ("lvpf", 0.47, -1.36, 0.0, 0),
        ("lvaf", 0.82,  1.94, 0.0, 0),
        ("rvsf", 0.73, -0.04, 0.0, 0),
        ("rvmod",0.63,  0.21, 0.0, 1),
    ]

    electrodes_folder = os.path.join(target_folder, "Electrodes")
    os.makedirs(electrodes_folder, exist_ok=True)
    all_indices = []

    for circle_name, a, r, m, v in circles_uvc:
        ab, rt, tm, tv = uvc_to_cobi(a, r, m, v)
        if circle_name == "rvsf": tv = 1
        dat_file   = os.path.join(target_folder, f"{circle_name}.dat")
        matches    = find_points_in_cylinder(csv_file, ab, rt, tm, tv, radius, thickness, dat_file)
        vtx_file   = os.path.join(electrodes_folder, f"{circle_name}.vtx")
        create_vtx_file(vtx_file, matches)
        all_indices.append(matches)

    combine_lists_and_write_dat(csv_file, all_indices, os.path.join(electrodes_folder, "combined_circles.dat"))
    for circle_name, *_ in circles_uvc:
        single_dat = os.path.join(target_folder, f"{circle_name}.dat")
        if os.path.exists(single_dat):
            os.remove(single_dat)

# --- search helper (verbatim) ----------------------------------------------
def find_points_in_cylinder(csv_file, base_ab, base_rt, base_tm, base_tv, radius, thickness, dat_file="output.dat"):
    matched_indices, flags = [], []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            ab  = float(row["ab"]); rt = float(row["rt"])
            tm  = float(row["tm"]); tv = float(row["tv"])
            if tv == base_tv and abs(tm-base_tm) <= thickness:
                for shift in (0, -1, +1):
                    if (ab-base_ab)**2 + (rt-base_rt+shift)**2 < radius**2:
                        matched_indices.append(idx); break
            flags.append(1 if idx in matched_indices else 0)
    with open(dat_file, "w") as out:
        for f in flags: out.write(f"{f}\n")
    return matched_indices
