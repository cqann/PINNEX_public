#!/usr/bin/env python

import os
from datetime import date

from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import testing
import numpy as np
from numpy import array as nplist
import matplotlib.pyplot as plt
from carputils import ep
from carputils.carpio import txt
import math
from scipy.signal import iirfilter, filtfilt
import pandas as pd  # <-- Import pandas
import random
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
        help="Duration of simulation (ms) (default is 50.)",
    )

    group.add_argument(
        "--simID",
        type=str,
        default=1,
        help="provide simID for which to compute the transmural ECG",
    )
    group.add_argument(
        "--random_seed",
        type=int,
        default=None,  # Default to None, meaning non-reproducible if not set
        help="Seed for the random number generator to ensure reproducible fibrosis patterns. (default: None)",
    )
    group.add_argument(
        "--visualize_ecg",
        action="store_true",  # Makes it a boolean flag (True if present, False otherwise)
        help="Display a plot of the computed 12-lead ECG after computation.",
    )
    group.add_argument(
        "--n_patches",
        type=int,
        default=None,  # Default to None, meaning non-reproducible if not set
        help="Display a plot of the computed 12-lead ECG after computation.",
    )
    group.add_argument("--mesh", type=str, default=None, help="provide mesh directory")

    return parser


# This sets up where we store the output of the experiment
def jobID(args):
    return args.simID


@tools.carpexample(parser, jobID)
def run(args, job):

    # ===================================
    # 1 : Defining the mesh:
    # ===================================
    mesh_folder = os.path.join(
        os.path.dirname(CALLER_DIR),
        "ml_heart",
        args.mesh + "_heart",
    )
    meshname = os.path.join(mesh_folder, "heart")
    # ===================================
    # 2 : Defining the ionic models & conductivities
    # ===================================
    pts_file = os.path.join(mesh_folder, "heart.pts")
    points = txt.read(pts_file)
    points = points[0].T
    mesh_sz = [float(max(k) - min(k)) / 1000 for k in points]
    setup_fibrois(meshname, args, mesh_sz, 500, mesh_folder)  # Fixa

    imp_reg = setup_ionic()
    g_reg = setup_gregions(args)

    # ===================================
    # 3 : Defining the stimulation:
    # ===================================
    # Define the full path to your vertex file
    # Define file paths explicitly
    lvaf_vtx = os.path.join(mesh_folder, "Electrodes", "lvaf.vtx")
    lvpf_vtx = os.path.join(mesh_folder, "Electrodes", "lvpf.vtx")
    lvsf_vtx = os.path.join(mesh_folder, "Electrodes", "lvsf.vtx")
    rvmod_vtx = os.path.join(mesh_folder, "Electrodes", "rvmod.vtx")
    rvsf_vtx = os.path.join(mesh_folder, "Electrodes", "rvsf.vtx")

    # Define the first stimulus list (stim_0) with -num_stim set to 5
    stim_0 = [
        "-num_stim",
        5,  # total number of stimuli
        "-stim[0].name",
        "lvaf",
        "-stim[0].elec.vtx_file",
        lvaf_vtx,
        "-stim[0].ptcl.duration",
        1.0,
        "-stim[0].ptcl.npls",
        1,
        "-stim[0].ptcl.start",
        10,
    ]

    # Define the other four stimuli, each with its own index [1] through [4]
    stim_1 = [
        "-stim[1].name",
        "lvpf",
        "-stim[1].elec.vtx_file",
        lvpf_vtx,
        "-stim[1].ptcl.duration",
        1.0,
        "-stim[1].ptcl.npls",
        1,
        "-stim[1].ptcl.start",
        10,
    ]

    stim_2 = [
        "-stim[2].name",
        "lvsf",
        "-stim[2].elec.vtx_file",
        lvsf_vtx,
        "-stim[2].ptcl.duration",
        1.0,
        "-stim[2].ptcl.npls",
        1,
        "-stim[2].ptcl.start",
        10,
    ]

    stim_3 = [
        "-stim[3].name",
        "rvmod",
        "-stim[3].elec.vtx_file",
        rvmod_vtx,
        "-stim[3].ptcl.duration",
        1.0,
        "-stim[3].ptcl.npls",
        1,
        "-stim[3].ptcl.start",
        25.3,
    ]

    stim_4 = [
        "-stim[4].name",
        "rvsf",
        "-stim[4].elec.vtx_file",
        rvsf_vtx,
        "-stim[4].ptcl.duration",
        1.0,
        "-stim[4].ptcl.npls",
        1,
        "-stim[4].ptcl.start",
        10,
    ]

    # Combine them into a single parameter list
    all_stims = stim_0 + stim_1 + stim_2 + stim_3 + stim_4

    # ecg = ["-phie_rec_ptf", os.path.join(torso_folder, "electrodes")]
    ecg = [
        "-phie_rec_ptf",
        os.path.join(mesh_folder, "leads_placement"),
        "-phie_rec_meth",
        "2",
    ]

    num_par = ["-dt", 50]  # us
    IO_par = ["-spacedt", 1, "-timedt", 1.0]  # ms  # ms

    cmd = tools.carp_cmd()
    cmd += [
        "-num_phys_regions",
        "1",
        "-phys_region[0].ptype",
        "2",  # 2 = PHYSREG_EIKONAL
        "-phys_region[0].num_IDs",
        "2",
        "-phys_region[0].ID[0]",
        "1",
        "-phys_region[0].ID[1]",
        "2",
        "-phys_region[0].name",
        "eikdomain",
    ]

    cmd += imp_reg
    cmd += g_reg
    cmd += all_stims
    cmd += num_par
    cmd += IO_par
    cmd += ecg

    simID = job.ID
    # Place the output folder inside the mesh folder ("Mean_0.82" folder)
    output_folder = os.path.join(mesh_folder, simID)
    print("Simulation output folder: ", output_folder)
    cmd += [
        "-meshname",
        meshname,
        "-tend",
        args.duration,
        "-simID",
        simID,
        "-eik_solve",
        "2",
        "-output_level",
        "2",
    ]

    job.carp(cmd, "ECG Tissue prototype")
    compute_tmECG(args.simID, job, args.webGUI, args.ID, args.visualize_ecg)
    compute_cv(meshname, args.simID)
    # compute_diff(mesh_folder,args.simID)


def setup_ionic():
    imp_reg = [
        "-num_imp_regions",
        1,
        "-imp_region[0].im",
        "MitchellSchaeffer",
        "-imp_region[0].num_IDs",
        2,  # Two element tags (IDs) for the same region
        "-imp_region[0].ID[0]",
        "1",
        "-imp_region[0].ID[1]",
        "2",
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
    CV_healthy = 600
    CV_ar = 1
    CV_factor = 0.25
    CV_fibrosis = CV_healthy * CV_factor
    if args.n_patches == 0:
        g_reg = [
            "-num_gregions",
            1,
            # region 0 => tag=1, healthy
            "-gregion[0].num_IDs",
            1,
            "-gregion[0].ID[0]",
            1,
            "-gregion[0].CV.vel",
            CV_healthy,
            "-gregion[0].CV.ar_t",
            CV_ar,
            "-gregion[0].CV.ar_n",
            CV_ar,
            "-gregion[0].g_bath",
            0.22,
        ]
    else:
        g_reg = [
            "-num_gregions",
            2,
            # region 0 => tag=1, healthy
            "-gregion[0].num_IDs",
            1,
            "-gregion[0].ID[0]",
            1,
            "-gregion[0].CV.vel",
            CV_healthy,
            "-gregion[0].CV.ar_t",
            CV_ar,
            "-gregion[0].CV.ar_n",
            CV_ar,
            "-gregion[1].num_IDs",
            1,
            "-gregion[1].ID[0]",
            2,
            "-gregion[1].CV.vel",
            CV_fibrosis,
            "-gregion[1].CV.ar_t",
            CV_ar,
            "-gregion[1].CV.ar_n",
            CV_ar,
            "-gregion[0].g_bath",
            0.22,
        ]

    return g_reg


def setup_fibrois(meshname, args, mesh_sz, dot_size, meshfolder):

    elems, etags, nelems = txt.read(meshname + "_original.elem")
    points, n_points = txt.read(
        os.path.join(meshfolder, "..", "1600_heart", "heart.pts")
    )
    print(f"Initializing random number generator with seed: {args.random_seed}")
    random.seed(args.random_seed)
    max_radius = 15000
    n_patches = args.n_patches
    fibrotic_points = []
    fasciles = []
    fasciles_file = os.path.join(
        meshfolder,
        "Electrodes",
        "combined_circles.dat",
    )
    with open(fasciles_file, "r") as f:
        for i, line in enumerate(f):
            if line.strip() == "1":
                fasciles.append(i)
    set_fasciles = set(fasciles)
    patches_generated = 0
    n_rounds = 0
    while patches_generated < n_patches:
        # Generate a random integer index between 0 and n_points-1 (inclusive)
        # This index corresponds to one of the points in your mesh.
        random_seed_point_index = random.randint(0, n_points - 1)
        x, y, z = points[random_seed_point_index]
        command = [
            "meshtool",
            "query",
            "idx",
            "-msh=" + meshname,
            f"-coord={x},{y},{z}",
            f"-thr={max_radius}",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            str_split = result.stdout.split("Vertex list:")
            if len(str_split) != 2:
                fibrotic_points = []
            else:
                new_fibrotic_points = [int(x) for x in str_split[1].strip().split(",")]
                set_new_fibrotic_points = set(new_fibrotic_points)
                common_vertex_set = set_fasciles.intersection(set_new_fibrotic_points)
                common_elements_found = sorted(list(common_vertex_set))
                common_elements_count = len(common_elements_found)
                if common_elements_count == 0:
                    fibrotic_points += new_fibrotic_points
                    patches_generated += 1
            n_rounds += 1
        except subprocess.CalledProcessError as e:
            print("Error executing meshtool command:")
            print(e.stderr)

        print(
            f"Patch {i+1}: Selected random seed point index: {random_seed_point_index}"
        )

    print("Took " + str(n_rounds))

    set_fibrotic_points = set(fibrotic_points)

    # Find the common elements by taking the intersection of the two sets
    common_vertex_set = set_fasciles.intersection(set_fibrotic_points)

    common_elements_found = sorted(list(common_vertex_set))

    common_elements_count = len(common_elements_found)

    length = len(fibrotic_points)
    print("Number of of fibrotic points found " + str(len(fibrotic_points)))
    print("Number of of common elements found " + str(common_elements_count))

    fibrotic_points_after_removal = sorted(
        list(set_fibrotic_points - common_vertex_set)
    )

    print("Number of of common points found " + str(common_elements_count))
    print(
        "Difference between orginal and new list  "
        + str(len(fibrotic_points_after_removal) - len(fibrotic_points))
    )

    fibrotic_tetra_set = set()
    vertex_to_tetra = (
        {}
    )  # This get vertex of a point a outputs the tetras connected to it
    for i, tet in enumerate(elems):
        for vertex in tet:
            vertex_to_tetra.setdefault(vertex, []).append(i)

    n_tissue_elems = len([x for x in etags if x != 0])

    fibrotic_tetra = set()
    for vertex in fibrotic_points:
        fibrotic_tetra.update(vertex_to_tetra.get(vertex, []))
    fibrotic_tetra = list(fibrotic_tetra)
    fibrotic_tetra_set = fibrotic_tetra_set.union(set(fibrotic_tetra))

    fasciles_tetra_set = set()
    fasciles_tetra = set()
    for vertex in common_elements_found:
        fasciles_tetra.update(vertex_to_tetra.get(vertex, []))
    fasciles_tetra = list(fasciles_tetra)
    fasciles_tetra_set = fasciles_tetra_set.union(set(fasciles_tetra))

    n_fibrotic_elements = len(fibrotic_tetra_set) - len(fasciles_tetra_set)
    print("Found " + str(n_fibrotic_elements))
    print("Vol fraction fibrosis " + str(n_fibrotic_elements / n_tissue_elems))

    etags[fibrotic_tetra] = 2
    # etags[fasciles_tetra] = 1

    txt.write(meshname + ".elem", elems, etags=etags)


def compute_cv(meshname, simID):
    elems, etags, nelems = txt.read(meshname + ".elem")
    points, n_points = txt.read(meshname + ".pts")

    print(f"Successfully read mesh data:")
    print(f"Number of elements (nelems): {nelems}")
    print(f"Number of points (n_points): {n_points}")

    # Conduction velocities for tissue types
    healthy_cv = 600.0  # Units typically mm/s or um/ms
    CV_factor = 0.25
    fibrotic_cv = healthy_cv * CV_factor
    # CV for 'other' elements is not defined, they won't contribute to the average CV calculation here

    print(f"\nHealthy tissue CV: {healthy_cv}")
    print(f"Fibrotic tissue CV: {fibrotic_cv}")

    vertex_to_tetra = {}
    for i, tet_element in enumerate(elems):  # i is the element index
        for vertex_idx in tet_element:  # vertex_idx is the point index in the element
            vertex_to_tetra.setdefault(vertex_idx, []).append(i)

    # Using definitions from your script:
    # Healthy if etag is 0 or 1
    # Fibrotic if etag is 2 or 3
    healthy_element_indices_list = [
        idx for idx in range(nelems) if etags[idx] == 0 or etags[idx] == 1
    ]
    fibrotic_element_indices_list = [
        idx for idx in range(nelems) if etags[idx] == 2 or etags[idx] == 3
    ]

    # Convert to sets for efficient lookup
    healthy_lookup_set = set(healthy_element_indices_list)
    fibrotic_lookup_set = set(fibrotic_element_indices_list)

    print(
        f"\nTotal 'healthy' elements defined (tags 0 or 1): {len(healthy_lookup_set)}"
    )
    print(
        f"Total 'fibrotic' elements defined (tags 2 or 3): {len(fibrotic_lookup_set)}"
    )

    point_tissue_connectivity_stats = []  # To store stats for each point

    for p_idx in range(n_points):
        num_healthy_connected = 0
        num_fibrotic_connected = 0
        num_other_connected = 0
        total_connected = 0

        # Sum of (CV * number_of_elements_of_that_type)
        # We only consider healthy and fibrotic for the CV sum based on provided CVs
        sum_cv_contributions = 0.0
        num_cv_contributing_elements = 0  # Denominator for average CV

        if p_idx in vertex_to_tetra:
            connected_element_indices = vertex_to_tetra[p_idx]
            total_connected = len(connected_element_indices)
            for elem_idx in connected_element_indices:
                if elem_idx in healthy_lookup_set:
                    num_healthy_connected += 1
                elif elem_idx in fibrotic_lookup_set:
                    num_fibrotic_connected += 1
                else:
                    num_other_connected += 1

            # Calculate sum of CVs from connected healthy/fibrotic elements
            sum_cv_contributions = (num_healthy_connected * healthy_cv) + (
                num_fibrotic_connected * fibrotic_cv
            )
            num_cv_contributing_elements = (
                num_healthy_connected + num_fibrotic_connected
            )

        avg_cv_at_point = 0.0
        if num_cv_contributing_elements > 0:
            avg_cv_at_point = sum_cv_contributions / num_cv_contributing_elements

        point_tissue_connectivity_stats.append(
            {
                "point_idx": p_idx,
                "healthy_connected": num_healthy_connected,
                "fibrotic_connected": num_fibrotic_connected,
                "other_connected": num_other_connected,
                "total_connected": total_connected,
                "average_cv": avg_cv_at_point,  # Added average CV for the point
            }
        )
    if point_tissue_connectivity_stats:  # Check if there's data to write
        df = pd.DataFrame(point_tissue_connectivity_stats)

        # Define the desired column order with 'average_cv' first
        column_order = [
            "point_idx",
            "average_cv",
            "healthy_connected",
            "fibrotic_connected",
        ]
        df = df[column_order]  # Reorder columns

        csv_filename = "point_cv_stats.csv"
        output_csv_full_path = os.path.join(simID, csv_filename)
        try:
            df.to_csv(output_csv_full_path, index=False, float_format="%.2f")
            print(f"\nPoint statistics successfully saved to: {output_csv_full_path}")
        except Exception as e:
            print(f"\nError saving CSV file: {e}")
    else:
        print("No point statistics data to save to CSV.")
    # return point_tissue_connectivity_stats
    return


def compute_tmECG(tmECG, job, webgui, idExp, visualize_ecg_flag=False):
    """
    Extract the epicardial signals, plot the raw electrode channels,
    compute standard 12‑lead ECG, and plot the computed ECG leads.
    """

    # ----------------------------
    # 1. Extract and read the raw epi data (9 channels)
    # ----------------------------
    extract_epi = [
        settings.execs.igbextract,
        "-l",
        "0-8",
        "-O",
        os.path.join(tmECG, "phie_epi.dat"),
        "-o",
        "ascii",
        os.path.join(tmECG, "phie_recovery.igb"),
    ]
    job.bash(extract_epi)

    file_path = os.path.join(tmECG, "phie_epi.dat")
    epi_data = np.loadtxt(file_path)
    if epi_data.ndim == 1:
        epi_data = epi_data.reshape(-1, 1)

    # Define the channel labels corresponding to the columns
    channel_labels = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "LL"]

    # Build a dictionary mapping each electrode to its signal.
    electrode_dict = {}
    for i, label in enumerate(channel_labels):
        electrode_dict[label] = epi_data[:, i]
    # ----------------------------
    # 3. Compute the standard 12-lead ECG
    # ----------------------------
    # Define RA, LA, and LL for easy reference.
    RA = electrode_dict["RA"]
    LA = electrode_dict["LA"]
    LL = electrode_dict["LL"]

    # Compute Wilson’s central terminal (V_W)
    V_W = (RA + LA + LL) / 3.0

    ecg_dict = {}
    # Limb leads:
    ecg_dict["I"] = LA - RA
    ecg_dict["II"] = LL - RA
    ecg_dict["III"] = LL - LA
    ecg_dict["aVR"] = RA - 0.5 * (LA + LL)
    ecg_dict["aVL"] = LA - 0.5 * (RA + LL)
    ecg_dict["aVF"] = LL - 0.5 * (RA + LA)
    # Precordial leads (V1–V6 relative to V_W):
    for vn in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        ecg_dict[vn] = electrode_dict[vn] - V_W
    lowcut = 0.05
    highcut = 60.0
    order = 3
    fs = 1000
    # Create a new dictionary to store the filtered results
    filtered_ecg_dict_iterative = {}

    for lead_name, lead_data in ecg_dict.items():
        print(f"  Filtering lead: {lead_name}...")

        # Check if data is valid before filtering
        if lead_data is not None and isinstance(lead_data, np.ndarray):
            if (
                lead_data.ndim == 1 and lead_data.size > order * 3
            ):  # filtfilt needs len > padlen (usually order*3)
                # Apply the filter to the current lead's data (which is 1D)
                filtered_signal = bessel_bandpass_filter_1d(
                    lead_data,
                    fs=fs,  # Pass the sampling frequency
                    lowcut=lowcut,
                    highcut=highcut,
                    order=order,
                )
                # Store the filtered signal in the new dictionary
                filtered_ecg_dict_iterative[lead_name] = filtered_signal
            elif lead_data.ndim != 1:
                print(
                    f"  Skipping {lead_name}: Data is not 1-dimensional (shape: {lead_data.shape})."
                )
                filtered_ecg_dict_iterative[lead_name] = (
                    lead_data  # Keep original if invalid shape
                )
            else:
                print(
                    f"  Skipping {lead_name}: Data length ({lead_data.size}) is too short for filter order ({order})."
                )
                filtered_ecg_dict_iterative[lead_name] = (
                    lead_data  # Keep original if too short
                )

        else:
            print(f"  Skipping {lead_name}: Data is None or not a NumPy array.")
            filtered_ecg_dict_iterative[lead_name] = (
                lead_data  # Keep original if invalid type
            )

    print("Iterative filtering complete.")

    # ----------------------------
    # 4. Plot computed ECG leads
    # ----------------------------
    ecg_dict = filtered_ecg_dict_iterative

    ecg_leads = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    print("Saving computed and filtered ECG leads to CSV...")
    # Define the standard order of leads for the CSV columns
    final_ecg_data = ecg_dict

    # Check if there is *any* valid data to save
    if not final_ecg_data or all(v is None for v in final_ecg_data.values()):
        print("  Warning: No valid filtered ECG data found to save.")
    else:
        # Determine number of samples and create time vector
        # Find the first valid lead to determine length
        num_samples = 0
        first_valid_key = None
        for lead in ecg_leads:
            if lead in final_ecg_data and final_ecg_data[lead] is not None:
                num_samples = len(final_ecg_data[lead])
                first_valid_key = lead
                break

        if num_samples > 0 and first_valid_key is not None:
            # Create time vector (assuming constant fs)
            time_vector = np.arange(num_samples) / fs

            # Prepare data dictionary for DataFrame construction
            data_for_csv = {}
            for lead in ecg_leads:
                signal = final_ecg_data.get(lead)  # Use .get() for safety
                # Ensure signal is valid and has the expected length
                if (
                    signal is not None
                    and isinstance(signal, np.ndarray)
                    and signal.size == num_samples
                ):
                    data_for_csv[lead] = signal
                else:
                    print(
                        f"  Warning: Data for lead {lead} is missing or invalid. Filling with NaN."
                    )
                    data_for_csv[lead] = np.full(
                        num_samples, np.nan
                    )  # Fill problematic columns with NaN

            # Create pandas DataFrame
            ecg_df = pd.DataFrame(data_for_csv)

            # Define output file path (save within the tmECG directory)
            output_filename = "ecg.csv"
            output_csv_path = os.path.join(tmECG, output_filename)

            # Save to CSV file
            # index=False prevents pandas from writing the DataFrame index as a column
            # float_format controls the precision of saved floats
            ecg_df.to_csv(output_csv_path, index=False, float_format="%.6f")
            print(f"  Successfully saved filtered ECG data to {output_csv_path}")

        else:
            print(
                "  Warning: Could not determine data length or find valid leads. Skipping CSV save."
            )
    # 3. Add the plotting block
    if visualize_ecg_flag and ecg_df is not None and time_vector is not None:
        print("Plotting computed ECG leads vs Healthy ECG...")

        # --- Load Healthy ECG Data ---
        # Construct the path relative to the tmECG directory
        healthy_file_path = os.path.join(
            os.path.dirname(tmECG.rstrip(os.sep)), "ecg_healthy.csv"
        )
        # Note: If tmECG is '/path/to/sim/output/', dirname gives '/path/to/sim',
        # so os.path.join combines it correctly. Added rstrip to handle potential trailing slash.

        healthy_ecg_df = None  # Initialize as None
        try:
            healthy_ecg_df = pd.read_csv(healthy_file_path)
            print(f"  Successfully loaded healthy ECG data from: {healthy_file_path}")
            # Basic check: Ensure loaded data is not empty and has similar length
            if healthy_ecg_df.empty:
                print(f"  Warning: Healthy ECG file {healthy_file_path} is empty.")
                healthy_ecg_df = None  # Treat as not loaded if empty
            elif len(healthy_ecg_df) != len(time_vector):
                print(
                    f"  Warning: Healthy ECG data length ({len(healthy_ecg_df)}) differs from computed ECG time vector length ({len(time_vector)}). Plotting based on time vector length."
                )
                # Data will be truncated or padded implicitly by matplotlib if lengths differ,
                # or you could explicitly truncate/align here if needed. Let's rely on plot length below.

        except FileNotFoundError:
            print(
                f"  Warning: Healthy ECG file not found at {healthy_file_path}. Plotting only computed ECG."
            )
        except Exception as e:
            print(
                f"  Error loading or reading healthy ECG file {healthy_file_path}: {e}. Plotting only computed ECG."
            )
            healthy_ecg_df = None  # Ensure it's None if any loading error occurred

        # --- Plotting Setup ---
        try:
            num_leads = len(ecg_leads)  # Assumes ecg_leads is defined earlier
            n_cols = 6
            n_rows = (num_leads + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(18, n_rows * 3),  # Slightly larger figure size for clarity
                sharex=True,
                squeeze=False,  # Ensure axes is always a 2D array even if n_rows/n_cols is 1
            )
            axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

            # --- Plotting Loop ---
            plot_count = 0  # To track how many subplots actually get data
            for i, lead in enumerate(ecg_leads):
                # Check if data exists and is not all NaN for computed ECG
                plot_computed = (
                    lead in ecg_df.columns and not ecg_df[lead].isnull().all()
                )
                # Check if data exists and is not all NaN for healthy ECG (and if df loaded)
                plot_healthy = (
                    healthy_ecg_df is not None
                    and lead in healthy_ecg_df.columns
                    and not healthy_ecg_df[lead].isnull().all()
                )

                if plot_computed or plot_healthy:
                    plot_count += 1
                    axes[i].set_title(lead)
                    axes[i].set_ylabel("Amplitude")  # Add units if known, e.g., "mV"
                    axes[i].grid(True)

                    # Determine max length based on available time vector
                    max_plot_len = len(time_vector)

                    if plot_computed:
                        # Plot computed data, ensuring length matches time vector slice
                        data_len = len(ecg_df[lead].values)
                        current_plot_len = min(max_plot_len, data_len)
                        axes[i].plot(
                            time_vector[:current_plot_len],
                            ecg_df[lead].values[:current_plot_len],
                            label="Computed",  # Add label for legend
                            linewidth=1.2,
                        )

                    if plot_healthy:
                        # Plot healthy data, ensuring length matches time vector slice
                        data_len_h = len(healthy_ecg_df[lead].values)
                        current_plot_len_h = min(max_plot_len, data_len_h)
                        axes[i].plot(
                            time_vector[:current_plot_len_h],
                            healthy_ecg_df[lead].values[:current_plot_len_h],
                            label="Healthy",  # Add label for legend
                            linestyle="--",  # Use dashed line for distinction
                            color="coral",  # Use a different color
                            linewidth=1.0,
                        )

                    # Add legend only if at least one trace was plotted
                    if plot_computed or plot_healthy:
                        axes[i].legend()

                else:
                    # Case where neither computed nor valid healthy data exists for this lead
                    axes[i].set_title(f"{lead} (No Data)")
                    axes[i].axis("off")  # Hide axes

            # --- Final Touches ---
            # Hide any unused subplots at the end
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            # Add x-axis label to the last row plots that are actually visible
            # Find the last visible axis index
            last_visible_ax_index = -1
            for ax_idx in range(len(axes) - 1, -1, -1):
                if axes[ax_idx].axison:
                    last_visible_ax_index = ax_idx
                    break
            # Add label to all visible axes in the last visible row
            if last_visible_ax_index != -1:
                last_visible_row = last_visible_ax_index // n_cols
                for k in range(n_cols):
                    idx = last_visible_row * n_cols + k
                    if idx < len(axes) and axes[idx].axison:
                        axes[idx].set_xlabel("Time (s)")

            # Set the main title
            if plot_count > 0:  # Only add title if something was plotted
                plt.suptitle("Computed vs. Healthy 12-Lead ECG")
            else:
                plt.suptitle("ECG Plotting (No data found to display)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect for suptitle

            # Save the plot instead of showing it interactively
            plot_filename = "computed_vs_healthy_ecg_plot.png"
            # Save in the parent directory alongside where ecg_healthy.csv is expected
            parent_dir = os.path.dirname(healthy_file_path)
            # Ensure parent_dir is valid if healthy_file_path construction failed, fallback to tmECG
            if not os.path.isdir(parent_dir):
                parent_dir = tmECG

            plot_output_path = os.path.join(parent_dir, plot_filename)

            try:
                plt.savefig(plot_output_path)
                print(f"  Plot saved to {plot_output_path}")
            except Exception as e:
                print(f"  Error saving plot to {plot_output_path}: {e}")

            # plt.show() # Keep commented out if running in non-interactive environment
            plt.close(fig)  # Close the figure to free memory

            print("  Plot generation finished.")

        except Exception as e:
            print(f"  An error occurred during the plotting process: {e}")
            # Ensure figure is closed if it exists, even if error happened mid-plot
            if "fig" in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)

    elif visualize_ecg_flag:
        # Handle cases where initial data (ecg_df or time_vector) is missing
        if ecg_df is None:
            print(
                "Skipping ECG plot: Computed ECG DataFrame (ecg_df) is missing or invalid."
            )
        elif time_vector is None:
            print("Skipping ECG plot: Time vector is missing or invalid.")
        else:
            # This case might occur if the flag is True but somehow df/vector became None between check and block start
            print(
                "Skipping ECG plot: Required data (ecg_df or time_vector) is unavailable."
            )


# --- Bessel Filter Function ---
# This function works correctly for single 1D time series arrays.
def bessel_bandpass_filter_1d(data, fs, lowcut=0.05, highcut=60.0, order=2):
    """
    Apply an Nth-order Bessel bandpass filter to a single time series (1D NumPy array).

    Args:
        data (np.ndarray): Input signal (1-dimensional array).
        fs (float): Sampling frequency of the data.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        order (int): Order of the Bessel filter.

    Returns:
        np.ndarray: Filtered signal (1-dimensional array).
    """
    # Input validation (optional but good practice)
    if not isinstance(data, np.ndarray):
        raise TypeError("Input 'data' must be a NumPy array.")
    if data.ndim != 1:
        # If you strictly want ONLY 1D, uncomment the next line
        # raise ValueError(f"Input 'data' must be 1-dimensional, but got shape {data.shape}")
        # Otherwise, this function *will* work on ND arrays by filtering the last axis.
        # For this use case (calling it per lead), data will be 1D.
        pass

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Add basic validation for frequency bounds
    low = max(low, 1e-9)  # Avoid Wn[0] <= 0
    high = min(high, 1 - 1e-9)  # Avoid Wn[1] >= 1
    if low >= high:
        raise ValueError(
            f"Bandpass filter frequency range is invalid: low={low*nyquist}Hz, high={high*nyquist}Hz. Lowcut must be less than Highcut and both must be less than Nyquist ({nyquist}Hz)."
        )

    # Design the filter
    b, a = iirfilter(
        N=order, Wn=[low, high], btype="band", ftype="bessel", analog=False, output="ba"
    )

    # Apply the filter using filtfilt (handles 1D case correctly)
    # axis=-1 works perfectly for 1D array
    filtered_data = filtfilt(b, a, data, axis=-1)

    return filtered_data


if __name__ == "__main__":
    run()
