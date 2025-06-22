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
import pandas as pd
import random
import subprocess
from plot_ecg import generate_ecg_plot_simple
import shutil

CALLER_DIR = os.path.dirname(__file__)


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
        default="1",
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
        default=0,  # Default to None, meaning non-reproducible if not set
        help="Display a plot of the computed 12-lead ECG after computation.",
    )
    group.add_argument(
        "--mesh", type=str, default="1600", help="provide mesh directory"
    )
    group.add_argument(
        "--heart", type=str, default="Mean", help="provide heart directory"
    )

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
        CALLER_DIR, "..", "Meshes", args.heart, args.mesh, "openCarp"
    )

    meshname = os.path.join(mesh_folder, "heart")
    # ===================================
    # 2 : Defining the ionic models & conductivities
    # ===================================
    pts_file = os.path.join(mesh_folder, "heart.pts")
    points = txt.read(pts_file)
    points = points[0].T
    setup_fibrois(meshname, args, mesh_folder)

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

    all_stims = stim_0 + stim_1 + stim_2 + stim_3 + stim_4

    ecg = [
        "-phie_rec_ptf",
        os.path.join(mesh_folder, "..", "..", "..", "leads_placement"),
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
        "3",
        "-phys_region[0].ID[0]",
        "1",
        "-phys_region[0].ID[1]",
        "2",
        "-phys_region[0].ID[2]",
        "3",
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
    output_folder = os.path.join(CALLER_DIR, simID)
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


def setup_ionic():
    imp_reg = [
        "-num_imp_regions",
        1,
        "-imp_region[0].im",
        "MitchellSchaeffer",
        "-imp_region[0].num_IDs",
        3,
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
    CV_healthy = 810
    CV_ar = 1
    CV_factor_core = 2.3
    CV_factor_border = 1.71
    CV_core = CV_healthy / CV_factor_core
    CV_border = CV_healthy / CV_factor_border
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
            3,
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
            CV_border,
            "-gregion[1].CV.ar_t",
            CV_ar,
            "-gregion[1].CV.ar_n",
            CV_ar,
            "-gregion[2].num_IDs",
            1,
            "-gregion[2].ID[0]",
            3,
            "-gregion[2].CV.vel",
            CV_core,
            "-gregion[2].CV.ar_t",
            CV_ar,
            "-gregion[2].CV.ar_n",
            CV_ar,
            "-gregion[0].g_bath",
            0.22,
        ]

    return g_reg


def setup_fibrois(meshname, args, meshfolder):
    orig_elem_file = meshname + "_original.elem"
    if not os.path.exists(orig_elem_file):
        shutil.copy(meshname + ".elem", orig_elem_file)
    elems, etags, nelems = txt.read(orig_elem_file)
    points, n_points = txt.read(os.path.join(meshfolder, "heart.pts"))
    if args.random_seed == None:
        random_seed = int(args.simID)
    else:
        random_seed = args.random_seed
    random.seed(random_seed)
    print(f"Initializing random number generator with seed: {random_seed}")
    max_radius = 15000
    small_radius = max_radius * 0.5
    n_patches = args.n_patches
    fibrotic_points = []
    core_fibrotic_points = []
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
                common_elements_count = len(common_vertex_set)
                if common_elements_count == 0:
                    fibrotic_points += new_fibrotic_points
                    patches_generated += 1
                    command = [
                        "meshtool",
                        "query",
                        "idx",
                        "-msh=" + meshname,
                        f"-coord={x},{y},{z}",
                        f"-thr={small_radius}",
                    ]
                    result = subprocess.run(
                        command, capture_output=True, text=True, check=True
                    )
                    str_split = result.stdout.split("Vertex list:")
                    new_core_fibrotic_points = [
                        int(x) for x in str_split[1].strip().split(",")
                    ]
                    core_fibrotic_points += new_core_fibrotic_points
            n_rounds += 1
        except subprocess.CalledProcessError as e:
            print("Error executing meshtool command:")
            print(e.stderr)

    print("Took " + str(n_rounds))

    set_fibrotic_points = set(fibrotic_points)
    common_vertex_set = set_fasciles.intersection(set_fibrotic_points)
    common_elements_found = sorted(list(common_vertex_set))
    common_elements_count = len(common_elements_found)

    print("Number of of common elements found " + str(common_elements_count))

    vertex_to_tetra = (
        {}
    )  # This get vertex of a point a outputs the tetras connected to it
    for i, tet in enumerate(elems):
        for vertex in tet:
            vertex_to_tetra.setdefault(vertex, []).append(i)

    n_tissue_elems = len([x for x in etags if x != 0])
    fibrotic_tetra_set = set()
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

    core_fibrotic_tetra = set()
    for vertex in core_fibrotic_points:
        core_fibrotic_tetra.update(vertex_to_tetra.get(vertex, []))
    core_fibrotic_tetra = list(core_fibrotic_tetra)

    n_fibrotic_elements = len(fibrotic_tetra_set)
    print("Found " + str(n_fibrotic_elements))
    vol_fraction_fibrosis = n_fibrotic_elements / n_tissue_elems
    print("Vol fraction fibrosis " + str(vol_fraction_fibrosis))

    etags[fibrotic_tetra] = 2
    etags[core_fibrotic_tetra] = 3
    output_file_path = os.path.join(CALLER_DIR, "fibrosis_vol_fractions_all_runs.txt")
    try:
        with open(output_file_path, "a") as f:  # 'a' for append mode
            f.write(f"{vol_fraction_fibrosis}\n")  # Write the fraction and a newline
        print(f"Successfully appended volume fraction to {output_file_path}")
    except Exception as e:
        print(f"Error appending volume fraction to {output_file_path}: {e}")
    txt.write(meshname + ".elem", elems, etags=etags)


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
    highcut = 60.0
    order = 2
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
                filtered_signal = bessel_lowpass_filter_1d(  # Use the new function
                    lead_data,
                    fs=fs,
                    highcut=highcut,  # Pass the highcut frequency
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
        healthy_ecg_file_path = "ecg_healthy.csv"

        # Ensure parent_dir is valid if healthy_file_path construction failed, fallback to tmECG
        plot_filename = "computed_vs_healthy_ecg_plot.png"
        # Save in the parent directory alongside where ecg_healthy.csv is expected
        parent_dir = os.path.dirname(healthy_ecg_file_path)
        # Ensure parent_dir is valid if healthy_file_path construction failed, fallback to tmECG
        if not os.path.isdir(parent_dir):
            parent_dir = tmECG

        generate_ecg_plot_simple(output_csv_path, healthy_ecg_file_path, ecg_leads)


# --- Bessel Filter Function ---
# --- Bessel Filter Function ---
# This function works correctly for single 1D time series arrays.
def bessel_lowpass_filter_1d(data, fs, highcut=60.0, order=2):
    """
    Apply an Nth-order Bessel lowpass filter to a single time series (1D NumPy array).

    Args:
        data (np.ndarray): Input signal (1-dimensional array).
        fs (float): Sampling frequency of the data.
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
    # For a lowpass filter, Wn is just the highcut frequency normalized by Nyquist
    normal_highcut = highcut / nyquist

    # Add basic validation for frequency bounds
    normal_highcut = min(normal_highcut, 1 - 1e-9)  # Avoid Wn >= 1
    if normal_highcut <= 0:
        raise ValueError(
            f"Lowpass filter cutoff frequency is invalid: highcut={highcut}Hz. Must be greater than 0 and less than Nyquist ({nyquist}Hz)."
        )

    # Design the filter
    # Change btype to 'low' and Wn to the single normalized highcut frequency
    b, a = iirfilter(
        N=order,
        Wn=normal_highcut,
        btype="low",
        ftype="bessel",
        analog=False,
        output="ba",
    )

    # Apply the filter using filtfilt (handles 1D case correctly)
    # axis=-1 works perfectly for 1D array
    # Check signal length is sufficient for the filter order
    if data.size <= order * 3:
        print(
            f"  Warning: Data length ({data.size}) is too short for filter order ({order}). Skipping filter."
        )
        return data  # Return original data if too short

    filtered_data = filtfilt(b, a, data, axis=-1)

    return filtered_data


if __name__ == "__main__":
    run()
