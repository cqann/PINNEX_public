import os
from datetime import date

import carputils
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import testing
import numpy as np
from numpy import array as nplist
import matplotlib.pyplot as plt
from carputils import ep
from carputils.carpio import txt
from carputils.carpio import meshtool

import math
import random
import re
import time
import subprocess


CALLER_DIR = os.getcwd()
EXAMPLE_DIR = os.path.dirname(__file__)


def parser():
    parser = tools.standard_parser()
    group = parser.add_argument_group("experiment specific options")

    group.add_argument(
        "--duration",
        type=float,
        default=50.0,
        help="Duration of simulation (ms).",
    )

    group.add_argument(
        "--ECG",
        type=str,
        default=None,
        help="Provide simID for which to compute the transmural ECG."
    )

    group.add_argument(
        "--simID",
        type=str,
        default=None,
        help="Custom simulation ID (optional)."
    )

    group.add_argument(
        "--seed",
        type=float,
        default=None,
        help="Custom seed"
    )

    group.add_argument(
        "--n_patches",
        type=int,
        default=None,
        help="Delay for the stimulation (ms).",
    )

    group.add_argument(
        "--fibrosis_vol_fraction",
        type=float,
        default=0.0,
        help="Volume fraction of fibrosis (0.0 to 1.0).",
    )

    return parser


def jobID(args):
    """
    Allow the user to specify a custom simID.
    If not provided, generate a default simID.
    """
    if args.simID:
        return args.simID
    else:
        today = date.today()
        return "{}_fibrosis_{}_vol{}".format(today.isoformat(), args.duration, int(args.fibrosis_vol_fraction * 100))


@tools.carpexample(parser, jobID)
def run(args, job):
    num_leads = 8

    # ---------------------------------------
    # 1 : Define the mesh geometry
    # ---------------------------------------
    x = 20  # (mm)
    y = 20  # (mm)
    z = 1   # (mm)
    res = 0.4

    geom = mesh.Block(
        centre=(0.0, 0.0, 0.0),
        size=(x, y, z),
        resolution=res,
        etype="tetra"
    )
    bath_size_factor = 0
    geom.set_bath(thickness=(x * bath_size_factor, y * bath_size_factor, 0), both_sides=True)

    # Regions:
    reg1 = mesh.BoxRegion(
        (-x / 2, -y / 2, -z / 2), (x / 2, y / 2, z / 2), tag=1
    )

    geom.add_region(reg1)

    geom.corner_at_origin()
    geom.set_fibres(0, 0, 0, 0)

    unique_dir = f"mesh_{int(time.time())}"
    meshname = mesh.generate(geom, rootdir=unique_dir)

    # Add fibrosis

    fibrosis_size = (2400, 0.4)
    fibrosis_size = (3500, 0.3)

    setup_fibrois(meshname, args, (x, y, z), fibrosis_size)  # 3500=1patch 2600=splash

    # Now, query the updated element tags for conduction
    _, new_etags, _ = txt.read(meshname + ".elem")
    all_tags = np.unique(new_etags)
    IntraTags = all_tags[all_tags != 0]  # for intracellular
    ExtraTags = all_tags.copy()          # for extracellular

    # ---------------------------------------
    # 2 : Define ionic models & conductivities
    # ---------------------------------------
    imp_reg = setup_ionic(args)
    g_reg = setup_gregions(args)

    # ---------------------------------------
    # 3 : Define the stimulation
    # ---------------------------------------
    stim = [
        "-num_stim", 1,
        "-stim[0].crct.type", 0,
        "-stim[0].pulse.strength", 1000.0,
        "-stim[0].ptcl.duration", 2.0,
        "-stim[0].ptcl.npls", 1,
        "-stim[0].ptcl.start", 0,
    ]

    # Example stimulus in corner
    stim_width_y = res  # (1 / 8) * y
    stim_y1 = (y - stim_width_y) / 2 * 1e3
    stim_y2 = (y + stim_width_y) / 2 * 1e3
    stim_width_x = res  # (1 / 16) * x * 1e3
    electrode = [
        "-stim[0].elec.p0[0]", 0,
        "-stim[0].elec.p1[0]", stim_width_x,
        "-stim[0].elec.p0[1]", stim_y1,
        "-stim[0].elec.p1[1]", stim_y2,
        "-stim[0].elec.p0[2]", 0,
        "-stim[0].elec.p1[2]", 1e3 * res * 2,
    ]

    # ---------------------------------------
    # 4 : Define extracellular recording sites
    # ---------------------------------------
    writeECGgrid(meshname, (x, y, z), num_leads * 2, res, bath_size_factor)
    ecg = ['-phie_rec_ptf', os.path.join(CALLER_DIR, 'ecg')]

    # ---------------------------------------
    # 5 : Define simulator options
    # ---------------------------------------
    num_par = ["-dt", 50]                # microseconds
    IO_par = ["-spacedt", 1, "-timedt", 1.0]  # output intervals (ms)

    cmd = tools.carp_cmd()
    cmd += imp_reg
    cmd += g_reg
    cmd += stim + electrode
    cmd += num_par + IO_par
    cmd += ecg
    # cmd += Src
    # cmd += tools.gen_physics_opts(ExtraTags=ExtraTags, IntraTags=IntraTags)
    # cmd += lat

    simID = job.ID
    cmd += ["-meshname", meshname, "-tend", args.duration, "-simID", simID, "-eik_solve", "2", "-output_level", "2"]
    if args.visualize:
        cmd += ["-gridout_i", 3, "-gridout_e", 3]

    cmd += [
        "-num_phys_regions", "1",
        "-phys_region[0].ptype", "2",  # 2 = PHYSREG_EIKONAL
        "-phys_region[0].num_IDs", "3",
        "-phys_region[0].ID[0]", "1",
        "-phys_region[0].ID[1]", "2",
        "-phys_region[0].ID[2]", "3",
        "-phys_region[0].name", "eikdomain"
    ]

    job.carp(cmd, "ECG Tissue prototype")
    if args.ECG is not None:
        # If asked, post-processing for ECG only
        compute_ECG(args.ECG, num_leads, job, args.ID)
        # return
    # If we want to visualize afterward:
    if args.visualize and not settings.platform.BATCH:
        geom_i = os.path.join(job.ID, os.path.basename(meshname) + "_i")
        data_vm = os.path.join(job.ID, "vm.igb")
        view_vm = os.path.join(EXAMPLE_DIR, "view_vm.mshz")
        aux_ecg = os.path.join(job.ID, "ecg.pts_t")

        recv_ptsFile = os.path.join(CALLER_DIR, "ecg.pts")
        recv_igbFile = os.path.join(job.ID, "phie_recovery.igb")
        txt.write(aux_ecg, recv_igbFile, ptsf=recv_ptsFile)

        job.meshalyzer(geom_i, data_vm, aux_ecg, view_vm)


def setup_ionic(args):
    imp_reg = [
        "-num_imp_regions",
        1,
        "-imp_region[0].im",
        "MitchellSchaeffer",
        "-imp_region[0].num_IDs",
        3,  # Two element tags (IDs) for the same region
        "-imp_region[0].ID[0]",
        "1",
        "-imp_region[0].ID[1]",
        "2",
        "-imp_region[0].ID[2]",
        "3",
        "-imp_region[0].im_param",
        "V_max=40.0,V_min=-86.2, tau_in = 0.3, tau_out=5.4,tau_open =80, tau_close=175",
    ]
    return imp_reg


def setup_gregions(args):
    """
    Healthy conduction: gi=0.174, ge=0.625
    Fibrotic conduction changes:
      - 80% reduction in transverse => g_it = 0.2 * gi
      - 50% reduction in longitudinal => g_il = 0.5 * gi
      => anisotropy ratio ~ 2.5x
    Region 6 = non-conductive => set conduction to 0.
    """
    CV_healthy = 810  # (mm/ms)
    CV_ar = 1 / 0.42  # (isotropic, 0.42 is the anisotropy ratio for healthy tissue)
    CV_ar = 1  # overwriting to keep isotropic

    g_reg = [
        "-num_gregions", (3 if args.fibrosis_vol_fraction > 0 or args.n_patches > 0 else 1),

        # region 0 => tag=1, healthy
        "-gregion[0].num_IDs", 1,
        "-gregion[0].ID[0]", 1,
        "-gregion[0].CV.vel", CV_healthy,
        "-gregion[0].CV.ar_t", CV_ar,
        "-gregion[0].CV.ar_n", CV_ar,
        "-gregion[0].g_bath", 0.22,
    ]

    if args.fibrosis_vol_fraction == 0 and args.n_patches == 0:
        return g_reg

    g_reg += [
        # region 2 => tag=3, fibrotic core
        "-gregion[1].num_IDs", 1,
        "-gregion[1].ID[0]", 2,
        "-gregion[1].CV.vel", CV_healthy / 2.3,
        "-gregion[1].CV.ar_t", CV_ar,
        "-gregion[1].CV.ar_n", CV_ar,


        # region 5 => tag=6, fibrotic border
        "-gregion[2].num_IDs", 1,
        "-gregion[2].ID[0]", 3,
        "-gregion[2].CV.vel", CV_healthy / 1.71,
        "-gregion[2].CV.ar_t", CV_ar,
        "-gregion[2].CV.ar_n", CV_ar,
    ]
    return g_reg


def setup_fibrois(meshname, args, mesh_sz, fiboris_size):
    """
    Sets up fibrotic regions in a mesh with a core and border zone.

    Fibrosis is generated by placing patches. Each patch has a dense core
    (tag 3) and a surrounding border zone (tag 2). The placement and size
    are determined by input arguments.
    """
    mesh_dim_x, mesh_dim_y, mesh_dim_z = mesh_sz
    elems, etags, nelems = txt.read(meshname + ".elem")
    dot_size, size_factor = fiboris_size  # Characteristic size of a patch in microns
    splash = args.n_patches == 0 and args.fibrosis_vol_fraction > 0

    # Global sets to accumulate tetrahedra indices for all patches
    overall_border_tetra = set()  # Tetra in the larger radius (max_radius) for any patch
    overall_core_tetra = set()   # Tetra in the smaller radius (small_radius) for any patch

    # Map vertex indices to tetrahedra indices
    vertex_to_tetra = {}
    for i, tet_vertices in enumerate(elems):
        for vertex_idx in tet_vertices:
            vertex_to_tetra.setdefault(vertex_idx, []).append(i)

    # Count initial number of tissue elements (etags != 0)
    n_tissue_elems = 0
    for tag_value in etags:
        if tag_value != 0:
            n_tissue_elems += 1

    patches_generated = 0
    random.seed(args.seed)  # Initialize random seed

    # Main loop to generate fibrotic patches
    while True:
        if not splash:  # Fixed number of patches mode
            if patches_generated >= args.n_patches:
                break
        else:  # Splash mode (target volume fraction)
            # Calculate current fibrotic volume (based on border zone tetra that are tissue)
            current_fibrotic_tissue_tetra_count = 0
            for tetra_idx in overall_border_tetra:
                if etags[tetra_idx] != 0:  # Check original tag
                    current_fibrotic_tissue_tetra_count += 1

            if n_tissue_elems == 0:  # Avoid division by zero if no tissue elements
                if args.fibrosis_vol_fraction > 0:
                    # If fibrosis is desired but no tissue exists, stop to prevent infinite loop.
                    break
            elif current_fibrotic_tissue_tetra_count / n_tissue_elems > args.fibrosis_vol_fraction:
                break

        # Generate a random seed point for the new patch (normalized coordinates)
        seed_norm_x = random.uniform(0, 1)
        seed_norm_y = random.uniform(0, 1)
        seed_norm_z = 0.5

        # Apply geometric constraint (e.g., cylindrical exclusion zone from original logic)
        if (seed_norm_x**2 + (seed_norm_y - 0.5)**2) <= 0.15:  # Check on normalized coordinates
            continue

        # Scale normalized coordinates to actual mesh coordinates for meshtool
        # Assuming mesh_sz (dims x,y,z) are in mm, and meshtool -coord expects microns
        actual_seed_x = seed_norm_x * mesh_dim_x * 1000
        actual_seed_y = seed_norm_y * mesh_dim_y * 1000
        actual_seed_z = seed_norm_z * mesh_dim_z * 1000

        # Define radii for the current patch's border and core zones
        # dot_size is the characteristic size of a patch, assumed to be in microns (like meshtool -thr)
        min_radius = dot_size * math.exp(-size_factor)
        max_radius = dot_size * math.exp(size_factor)

        current_patch_radius = random.uniform(min_radius, max_radius)  # Random size for the patch)

        current_patch_core_radius = current_patch_radius * 0.5  # Core radius is 50% of the patch's max radius

        # --- Query meshtool for vertices within the border zone (larger radius) ---
        border_points_this_patch = []
        cmd_border = ["meshtool", "query", "idx", f"-msh={meshname}",
                      f"-coord={actual_seed_x},{actual_seed_y},{actual_seed_z}",
                      f"-thr={current_patch_radius}"]
        try:
            result = subprocess.run(cmd_border, capture_output=True, text=True, check=True)
            str_split = result.stdout.split("Vertex list:")
            # Ensure vertex list exists and is not empty before parsing
            if len(str_split) == 2 and str_split[1].strip():
                border_points_this_patch = [int(p) for p in str_split[1].strip().split(",")]
        except subprocess.CalledProcessError as e:
            print(f"Warning: meshtool error for border zone query: {e.stderr.strip()}")
            continue  # Skip this patch attempt if meshtool fails

        # --- Query meshtool for vertices within the core zone (smaller radius) ---
        core_points_this_patch = []
        cmd_core = ["meshtool", "query", "idx", f"-msh={meshname}",
                    f"-coord={actual_seed_x},{actual_seed_y},{actual_seed_z}",
                    f"-thr={current_patch_core_radius}"]
        try:
            result = subprocess.run(cmd_core, capture_output=True, text=True, check=True)
            str_split = result.stdout.split("Vertex list:")
            if len(str_split) == 2 and str_split[1].strip():
                core_points_this_patch = [int(p) for p in str_split[1].strip().split(",")]
        except subprocess.CalledProcessError as e:
            print(f"Warning: meshtool error for core zone query: {e.stderr.strip()}")
            continue  # Skip this patch attempt

        # Convert vertex lists to sets of tetrahedra indices for the current patch
        patch_border_tetra = set()
        for v_idx in border_points_this_patch:
            patch_border_tetra.update(vertex_to_tetra.get(v_idx, []))

        patch_core_tetra = set()
        for v_idx in core_points_this_patch:
            patch_core_tetra.update(vertex_to_tetra.get(v_idx, []))

        # Add this patch's tetrahedra to the overall accumulation sets
        if not patch_border_tetra and not patch_core_tetra and args.n_patches > 0:  # if patch is empty, don't count it unless in splash
            # if we want N patches, an empty patch is a failed attempt for that N, so we shouldn't inc patches_generated
            # and should retry to get a non-empty patch.
            # However, if an area is truly sparse, this could loop. For now, count the attempt.
            pass  # or continue to retry for a non-empty patch, depends on desired behavior for empty meshtool results

        overall_border_tetra.update(patch_border_tetra)
        overall_core_tetra.update(patch_core_tetra)

        patches_generated += 1
    print("Generated {} patches.".format(patches_generated))
    # --- Update etags for fibrotic regions ---
    # Tag 2 for border zone, Tag 3 for core zone.
    # Core (tag 3) should override Border (tag 2) if a tetrahedron is in both.
    # This is achieved by first tagging all border elements, then all core elements.
    # Only modify etags of elements that were originally tissue (original tag != 0).

    for tetra_idx in overall_border_tetra:
        if etags[tetra_idx] != 0:  # Check if it's a tissue element
            etags[tetra_idx] = 2  # Mark as border zone

    for tetra_idx in overall_core_tetra:
        if etags[tetra_idx] != 0:  # Check if it's a tissue element (could have been marked 2 already)
            etags[tetra_idx] = 3  # Mark (or re-mark) as core zone

    # Write the updated element data (elems and modified etags)
    txt.write(meshname + ".elem", elems, etags=etags)


def writeECGgrid(meshname, mesh_sz, num_points, res, bath_size_factor):
    import subprocess
    import numpy as np
    from carputils.carpio import txt

    x_len, y_len, z_len = mesh_sz

    # We place the 'virtual electrodes' on a circle around the block’s top face:
    x_bath_len = x_len * (1 + 2 * bath_size_factor)
    y_bath_len = y_len * (1 + 2 * bath_size_factor)
    r = min(0.5 * x_bath_len, 0.5 * y_bath_len) * 0.9

    node_indices = []

    electrodes = [
        [0, -y_len / 3, z_len],
        [x_len / 3, -y_len / 3, 0],
        [2 * x_len / 3, -y_len / 3, z_len],
        [x_len, -y_len / 3, 0],

        [-x_len / 3, 0, z_len],
        [-x_len / 3, y_len / 3, 0],
        [-x_len / 3, 2 * y_len / 3, z_len],
        [-x_len / 3, y_len, 0],

        [x_len, 4 * y_len / 3, 0],
        [x_len / 3, 4 * y_len / 3, z_len],
        [0, 4 * y_len / 3, 0],
        [2 * x_len / 3, 4 * y_len / 3, z_len],

        [4 * x_len / 3, y_len / 3, 0],
        [4 * x_len / 3, y_len / 3, z_len],
        [4 * x_len / 3, 2 * y_len / 3, 0],
        [4 * x_len / 3, 2 * y_len / 3, z_len],
    ]

    write_coordinates = True
    pts = []

    # Generate the 'desired' coordinates, then for each coordinate run meshtool query idx:
    for x, y, z in electrodes:

        # Convert to microns for meshtool (CARP uses microns in .pts/.elem),
        # or keep in mind your mesh might be in mm already. Adjust if needed.
        query_x = x * 1e3
        query_y = y * 1e3
        query_z = z * 1e3

        if write_coordinates:
            pts.append([query_x, query_y, query_z])
            continue

        command = [
            "meshtool", "query", "idx",
            f"-msh={meshname}",
            f"-coord={query_x},{query_y},{query_z}",
            f"-thr={res * 1e3 * 0.99}"  # small threshold
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # result.stdout might look like:
            #   "Found 1 vertices within threshold=0.1\nVertex list:\n1234\n"
            lines = result.stdout.split("Vertex list:")
            if len(lines) == 2:
                # The line after 'Vertex list:' has the actual indices, possibly separated by commas
                idx_line = lines[1].strip().replace(",", " ")
                # Just take the first index if more than one is found
                found_indices = [int(v) for v in idx_line.split()]
                if found_indices:
                    node_indices.append(found_indices[0])
                else:
                    # If none found, push a sentinel like -1
                    node_indices.append(-1)
            else:
                node_indices.append(-1)
        except subprocess.CalledProcessError as e:
            print("Error executing meshtool query idx:", e.stderr)
            node_indices.append(-1)

    # Write the node indices to "ecg.pts". Each line in ecg.pts is a single index
    # You can also store them in a single column if you prefer
    if write_coordinates:
        pts_array = np.array(pts)
        txt.write(os.path.join(CALLER_DIR, "ecg.pts"), pts_array)
    else:
        node_indices = np.array(node_indices, dtype=int).reshape(-1, 1)
        txt.write("ecg.pts", node_indices)


def compute_ECG(ECG, num_leads, job, idExp, meshname=None):
    import re
    from carputils import settings
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Determine the path for the corresponding healthy simulation
    healthy_ECG = re.sub(r"vol\d+", "vol0", ECG)

    fibrotic_leads = []  # List to store fibrotic simulation leads
    healthy_leads = []   # List to store healthy simulation leads

    # Define paths to the source IGB files
    fibrotic_igb_path = os.path.join(ECG, "phie_recovery.igb")
    healthy_igb_path = os.path.join(healthy_ECG, "phie_recovery.igb")

    # --- Process each lead ---
    for i in range(num_leads):
        vtx_endo = str(i)
        vtx_epi = str(i + num_leads)

        # Temporary file paths for extracted data (will be overwritten/deleted)
        # Using temporary directory might be cleaner if permissions allow
        temp_dir = ECG  # Or use tempfile module for better practice
        endo_file_fib = os.path.join(temp_dir, f"temp_phie_fib_endo_{i}.dat")
        epi_file_fib = os.path.join(temp_dir, f"temp_phie_fib_epi_{i}.dat")
        endo_file_healthy = os.path.join(temp_dir, f"temp_phie_healthy_endo_{i}.dat")
        epi_file_healthy = os.path.join(temp_dir, f"temp_phie_healthy_epi_{i}.dat")

        # --- Fibrotic Simulation ---
        # Extract endocardial potential
        cmd_extract_endo_fib = [
            settings.execs.igbextract, "-l", vtx_endo, "-O", endo_file_fib,
            "-o", "ascii", fibrotic_igb_path
        ]
        job.bash(cmd_extract_endo_fib)

        # Extract epicardial potential
        cmd_extract_epi_fib = [
            settings.execs.igbextract, "-l", vtx_epi, "-O", epi_file_fib,
            "-o", "ascii", fibrotic_igb_path
        ]
        job.bash(cmd_extract_epi_fib)

        # Read traces and calculate lead ECG
        endo_trace_fib = txt.read(endo_file_fib)
        epi_trace_fib = txt.read(epi_file_fib)
        fibrotic_leads.append(endo_trace_fib - epi_trace_fib)

        # --- Healthy Simulation ---
        # Extract endocardial potential
        cmd_extract_endo_healthy = [
            settings.execs.igbextract, "-l", vtx_endo, "-O", endo_file_healthy,
            "-o", "ascii", healthy_igb_path
        ]
        job.bash(cmd_extract_endo_healthy)

        # Extract epicardial potential
        cmd_extract_epi_healthy = [
            settings.execs.igbextract, "-l", vtx_epi, "-O", epi_file_healthy,
            "-o", "ascii", healthy_igb_path
        ]
        job.bash(cmd_extract_epi_healthy)

        # Read traces and calculate lead ECG
        endo_trace_healthy = txt.read(endo_file_healthy)
        epi_trace_healthy = txt.read(epi_file_healthy)
        healthy_leads.append(endo_trace_healthy - epi_trace_healthy)

        try:
            os.remove(endo_file_fib)
            os.remove(epi_file_fib)
            os.remove(endo_file_healthy)
            os.remove(epi_file_healthy)
        except OSError as e:
            print(f"Warning: Could not remove temporary file: {e}")

    # --- Combine leads and save to CSV ---
    # Convert lists to NumPy arrays and stack them column-wise
    fibrotic_leads_array = np.column_stack(fibrotic_leads)
    healthy_leads_array = np.column_stack(healthy_leads)

    # Define output CSV file paths
    fibrotic_csv_path = os.path.join(ECG, "fibrotic_ECG.csv")
    healthy_csv_path = os.path.join(healthy_ECG, "healthy_ECG.csv")

    # Save arrays to CSV files
    np.savetxt(fibrotic_csv_path, fibrotic_leads_array, delimiter=",", fmt="%.6f")
    np.savetxt(healthy_csv_path, healthy_leads_array, delimiter=",", fmt="%.6f")

    print(f"Fibrotic ECG leads saved to: {fibrotic_csv_path}")
    print(f"Healthy ECG leads saved to: {healthy_csv_path}")

    plot = False
    if plot:
        max_rows = 4
        ncols = math.ceil(num_leads / max_rows)
        nrows = min(num_leads, max_rows)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i in range(num_leads):
            ax = axes[i]
            ax.plot(fibrotic_leads[i], label="Fibrotic", color='red', linewidth=2)
            ax.plot(healthy_leads[i], label="Healthy", color='black', linestyle='--', linewidth=2)
            ax.set_ylabel("Transmural ECG")
            ax.set_title(f"Lead {i}")
            ax.legend()

        for j in range(num_leads, len(axes)):
            axes[j].axis("off")

        for ax in axes[-ncols:]:
            ax.set_xlabel("Time")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run()
