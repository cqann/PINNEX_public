import os
from datetime import date

import shutil

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
        "--sourceModel", type=str, default="monodomain", help="Use mono- or bidomain."
    )

    group.add_argument(
        "--ECG",
        type=str,
        default=None,
        help="Provide simID for which to compute the transmural ECG."
    )

    group.add_argument(
        "--fibrosis_vol_fraction",
        type=float,
        default=0.0,
        help="Factor to modify fibrosis prevalance."
    )
    group.add_argument(
    "--stim_delay",
    type=float,
    default=0.0,
    help="Delay (ms) before the stimulation starts."
    )
    group.add_argument(
        "--simID",
        type=str,
        default=None,
        help="Custom simulation ID (optional)."
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
    num_leads = 12
    model = "AlievPanfilov"
    

    # ---------------------------------------
    # 1 : Define the mesh geometry
    # ---------------------------------------
    x = 4  # (mm)
    y = 4  # (mm)
    z = 0.6   # (mm)
    res = 0.2

    geom = mesh.Block(
        centre=(0.0, 0.0, 0.0),
        size=(x, y, z),
        resolution=res,
        etype="tetra"
    )
    bath_size_factor = 0.5
    geom.set_bath(thickness=(x * bath_size_factor, y * bath_size_factor, 0), both_sides=True)

    # Regions:
    reg1 = mesh.BoxRegion(
        (-x / 2, -y / 2, -z / 2), (x / 2, y / 2, z / 2), tag=1
    )

    geom.add_region(reg1)

    geom.corner_at_origin()
    geom.set_fibres(0, 0, 0, 0)

    #unique_dir = f"mesh_{int(time.time())}"
    #meshname = mesh.generate(geom, rootdir=unique_dir)
    # Set a simpler directory name
    # Define the directory for saving the mesh
    mesh_dir = "mesh"
    os.makedirs(mesh_dir, exist_ok=True)
    if os.path.exists(mesh_dir):
        shutil.rmtree(mesh_dir)

    # Create a fresh mesh directory
    os.makedirs(mesh_dir, exist_ok=True)


    

    # Generate the mesh inside mesh_simulation (returns the generated mesh's path)
    meshname = mesh.generate(geom, rootdir=mesh_dir)
    #unique_dir = f"mesh_{int(time.time())}"
    #meshname = mesh.generate(geom, rootdir=unique_dir)

    # Print the generated mesh path to verify
    print(f"Mesh saved at: {meshname}")

    # Add fibrosis
    setup_fibrois(meshname, args, (x, y, z), 100)

    # Now, query the updated element tags for conduction
    _, new_etags, _ = txt.read(meshname + ".elem")
    all_tags = np.unique(new_etags)
    IntraTags = all_tags[all_tags != 0]  # for intracellular
    ExtraTags = all_tags.copy()          # for extracellular

    # ---------------------------------------
    # 2 : Define ionic models & conductivities
    # ---------------------------------------
    imp_reg = setup_ionic(model)
    g_reg = setup_gregions()

    # ---------------------------------------
    # 3 : Define the stimulation
    # ---------------------------------------
    stim = [
    "-num_stim", 1,
    "-stim[0].crct.type", 0,
    "-stim[0].pulse.strength", 200.0,
    "-stim[0].ptcl.duration", 2.0,
    "-stim[0].ptcl.npls", 1,
    "-stim[0].ptcl.start", args.stim_delay,  # Use the user-specified delay here
    ]

    # Example stimulus in corner
    stim_width_y = (1 / 2) * y
    stim_y1 = (y - stim_width_y) / 2 * 1e3
    stim_y2 = (y + stim_width_y) / 2 * 1e3
    stim_width_x = (1 / 16) * x * 1e3
    electrode = [
        "-stim[0].elec.p0[0]", 0,
        "-stim[0].elec.p1[0]", stim_width_x,
        "-stim[0].elec.p0[1]", stim_y1,
        "-stim[0].elec.p1[1]", stim_y2,
        "-stim[0].elec.p0[2]", 0,
        "-stim[0].elec.p1[2]", 1e3 * z,
    ]

    # ---------------------------------------
    # 4 : Define extracellular recording sites
    # ---------------------------------------
    writeECGgrid((x, y, z), num_leads * 2, bath_size_factor)
    ecg = ['-phie_rec_ptf', os.path.join(CALLER_DIR, 'ecg')]

    # ---------------------------------------
    # 5 : Define simulator options
    # ---------------------------------------
    Src = ep.model_type_opts(args.sourceModel)
    num_par = ["-dt", 100]                # microseconds
    IO_par = ["-spacedt", 0.1, "-timedt", 1.0]  # output intervals (ms)

    cmd = tools.carp_cmd()
    cmd += imp_reg
    cmd += g_reg
    cmd += stim + electrode
    cmd += num_par + IO_par
    cmd += ecg
    cmd += Src
    cmd += tools.gen_physics_opts(ExtraTags=ExtraTags, IntraTags=IntraTags)

    simID = job.ID
    cmd += ["-meshname", meshname, "-tend", args.duration, "-simID", simID]
    if args.visualize:
        cmd += ["-gridout_i", 3, "-gridout_e", 3]

    if model == "AlievPanfilov":
        cmd += [
            "-num_gvecs", 1,
            "-gvec[0].name", "V",
            "-gvec[0].units", "mV",
            "-gvec[0].imp", "AlievPanfilov",
            "-gvec[0].ID[0]", "V",
            "-gvec[0].bogus", 0
        ]

    job.carp(cmd, "ECG Tissue prototype")
    if args.ECG is not None:
        # If asked, post-processing for ECG only
        compute_ECG(args.ECG, num_leads, job, args.ID)
        return
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

        geom_phie = os.path.join(CALLER_DIR, "ecg.pts")
        data_phie = os.path.join(job.ID, "phie_recovery.igb")
        view_phie = os.path.join(EXAMPLE_DIR, "view_phie_recovery.mshz")
        job.meshalyzer(geom_phie, data_phie, view_phie)

        if "bidomain" in args.sourceModel:
            geom_e = os.path.join(job.ID, os.path.basename(meshname) + "_e")
            data_e = os.path.join(job.ID, "phie.igb")
            view_e = os.path.join(EXAMPLE_DIR, "view_phie.mshz")
            job.meshalyzer(geom_e, data_e, view_e)


def setup_ionic(model="ttpn"):
    """Configure ionic parameters for either 'ttpn' or 'courtemanche' models."""
    if model == "ttpn":
        im_name = "tenTusscherPanfilov"
        fibrotic_params = "GNa*0.4,GK1*0.5,GCaL*0.5,GKr*0.6"
    elif model == "courtemanche":
        im_name = "Courtemanche"
        fibrotic_params = "GNa*0.4,GK1*0.5,GCaL*0.5"
    elif model == "AlievPanfilov":
        im_name = "AlievPanfilov"
        fibrotic_params = "mu1*0.4,mu2*0.5"
    else:
        return "model not implemented"

    imp_reg = ["-num_imp_regions", 3]
    imp_names = ["healthy", "fibrotic", "non-conductive"]
    for i in range(3):
        imp_reg.extend([
            f"-imp_region[{i}].im", im_name,
            f"-imp_region[{i}].num_IDs", 1,
            f"-imp_region[{i}].ID[0]", str(i + 1),
            f"-imp_region[{i}].name", imp_names[i],
        ])
        if i == 1:  # Region 3 (fibrotic)
            imp_reg.extend(["-imp_region[1].im_param", fibrotic_params])
        elif i == 0 and model == "AlievPanfilov":
            imp_reg.extend(["-imp_region[0].im_sv_dumps", "V"])
    return imp_reg


def setup_gregions():
    """
    Healthy conduction: gi=0.174, ge=0.625
    Fibrotic conduction changes:
      - 80% reduction in transverse => g_it = 0.2 * gi
      - 50% reduction in longitudinal => g_il = 0.5 * gi
      => anisotropy ratio ~ 2.5x
    Region 6 = non-conductive => set conduction to 0.
    """

    g_il_healthy = 1.0124
    g_it_healthy = 0.1122
    g_in_healthy = 0.1122
    g_el_healthy = 0.5062
    g_et_healthy = 0.1496
    g_en_healthy = 0.1496

    # Fibrotic conduction (region 3):
    gi_l_fib = g_il_healthy * 0.5**2   # 50% for longitudinal
    gi_t_fib = g_it_healthy * 0.2**2  # 80% reduction for transverse
    # We’ll just do same for 'n' direction if 3D => gi_n_fib = gi_t_fib or similar
    gi_n_fib = g_in_healthy

    # Non-conductive region (region 6):
    gi_zero = 1e-10
    ge_zero = 1e-10
    #gi_zero = 0
    #ge_zero = 0

    g_reg = [
        "-num_gregions", 3,

        # region 0 => tag=1, healthy
        "-gregion[0].num_IDs", 1,
        "-gregion[0].ID[0]", 1,
        "-gregion[0].g_il", g_il_healthy,
        "-gregion[0].g_it", g_it_healthy,
        "-gregion[0].g_in", g_in_healthy,
        "-gregion[0].g_el", g_el_healthy,
        "-gregion[0].g_et", g_et_healthy,
        "-gregion[0].g_en", g_en_healthy,

        # region 2 => tag=3, fibrotic myocytes
        "-gregion[1].num_IDs", 1,
        "-gregion[1].ID[0]", 2,
        "-gregion[1].g_il", gi_zero,
        "-gregion[1].g_it", gi_zero,
        "-gregion[1].g_in", gi_zero,
        "-gregion[1].g_el", gi_zero,  # often extracellular is less changed
        "-gregion[1].g_et", gi_zero,
        "-gregion[1].g_en", gi_zero,

        # region 5 => tag=6, non-conductive ECM
        "-gregion[2].num_IDs", 1,
        "-gregion[2].ID[0]", 3,
        "-gregion[2].g_il", gi_zero,
        "-gregion[2].g_it", gi_zero,
        "-gregion[2].g_in", gi_zero,
        "-gregion[2].g_el", ge_zero,
        "-gregion[2].g_et", ge_zero,
        "-gregion[2].g_en", ge_zero,
    ]
    return g_reg



def setup_fibrois(meshname, args, mesh_sz, dot_size):
    """
    Setup fibrosis in the tissue mesh by labeling a box region as fibrotic.
    The fibrotic region is defined as a box covering:
      - x: a fraction (args.fibrosis_vol_fraction) of the total x dimension,
           centered on the middle of the x-axis,
      - y: the full range,
      - z: the full range.
    Assumes:
      - Mesh elements and tags are read from meshname+".elem"
      - Node coordinates are available in meshname+".pts"
      - A txt module is available with read and write functions.
    """
    # Mesh dimensions (in mm) and read mesh data
    x, y, z = mesh_sz
    elems, etags, nelems = txt.read(meshname + ".elem")

    # Build mapping from vertex to tetrahedra indices
    vertex_to_tetra = {}
    for i, tet in enumerate(elems):
        for vertex in tet:
            vertex_to_tetra.setdefault(vertex, []).append(i)

    # Count only tissue elements (nonzero etags)
    n_tissue_elems = len([tag for tag in etags if tag != 0])
    print("Number of tissue tetrahedra:", n_tissue_elems)
    target_count = int(args.fibrosis_vol_fraction * n_tissue_elems)

    # ---------------------------
    # Define the fibrosis box region
    # ---------------------------
    # Read node coordinates from the pts file.
    node_coords_all = txt.read(meshname + ".pts")

    
    # Remove header if needed.
    # We expect each coordinate entry to be a list/tuple of 3 numbers.
    # If the first entry does not have length 3, assume it's a header and remove it.
    

    if len(node_coords_all[0]) != 3:
        node_coords_all = node_coords_all[0:]
    node_coords = node_coords_all[0]
    
    # Convert mesh dimensions to meshtool units (e.g., microns)
    x_total = x * 1000.0
    y_total = y * 1000.0
    z_total = z * 1000.0

    # Compute the box boundaries in x-direction:
    # The box width is the fraction (f) of the total x-length and is centered.
    f = args.fibrosis_vol_fraction  # e.g., 0.3 means 30% of the x-length
    box_width = f * x_total
    x_center = x_total / 2.0
    x_min = x_center - (box_width / 2.0)
    x_max = x_center + (box_width / 2.0)

    # The fibrotic region spans the entire y and z directions
    y_min, y_max = 0.0, y_total
    z_min, z_max = 0.0, z_total

    # -------------------------------------------
    # Identify tetrahedra with at least one vertex inside the defined fibrosis box
    # -------------------------------------------
    fibrotic_tetra_set = set()
    print(node_coords)
    for v_idx, (vx, vy, vz) in enumerate(node_coords):
        if (x_min <= vx <= x_max) and (y_min <= vy <= y_max) and (z_min <= vz <= z_max):
            if v_idx in vertex_to_tetra:
                fibrotic_tetra_set.update(vertex_to_tetra[v_idx])

    # Filter to include only tissue tetrahedra (etags != 0)
    fibrotic_tetras = [i for i in fibrotic_tetra_set if etags[i] != 0]

    # Shuffle and trim the list to the target count if necessary
    random.shuffle(fibrotic_tetras)
    if len(fibrotic_tetras) > target_count:
        fibrotic_tetras = fibrotic_tetras[:target_count]

    # Calculate actual count and percent achieved
    actual_count = len(fibrotic_tetras)
    percent_achieved = (actual_count / target_count * 100) if target_count > 0 else 0.0

    # -------------------------------------------------------
    # Split fibrotic tetrahedra into two groups:
    # half remodeling (tag 2) and half ECM (tag 3)
    # -------------------------------------------------------
    midpoint = len(fibrotic_tetras) // 2
    fibrotic_remodelled = fibrotic_tetras[:midpoint]
    fibrotic_ecm = fibrotic_tetras[midpoint:]
    for idx in fibrotic_remodelled:
        etags[idx] = 2
    for idx in fibrotic_ecm:
        etags[idx] = 3

    # Write the updated mesh with new tags
    txt.write(meshname + ".elem", elems, etags=etags)

    # Debug message reporting the result
    print(f"Debug: Fibrotic region (box) contains {actual_count} tissue tetrahedra, "
          f"which is {percent_achieved:.2f}% of the intended {target_count} elements.")
def writeECGgrid(mesh_sz, num_points, bath_size_factor):
    # Simple line of ECG points
    x_len, y_len, z_len = mesh_sz

    x_bath_len = x_len * (1 + 2 * bath_size_factor)
    y_bath_len = y_len * (1 + 2 * bath_size_factor)
    r = min(0.5 * x_bath_len, 0.5 * y_bath_len) * 0.9

    pts = []
    for i in range(num_points):
        x = x_len / 2 + r * np.cos(2 * np.pi * i / num_points)
        y = y_len / 2 + r * np.sin(2 * np.pi * i / num_points)
        z = z_len
        pts.append([x * 1e3, y * 1e3, z * 1e3])

    pts_array = np.array(pts)
    txt.write(os.path.join(CALLER_DIR, "ecg.pts"), pts_array)

def compute_ECG(ECG, num_leads, job, idExp, plot=False):
    """
    Extracts and optionally plots ECG leads for both the current (fibrotic) simulation and
    the healthy simulation. Saves them in a single CSV file with one row per time step.

    Parameters:
        ECG (str): Path to the ECG data directory.
        num_leads (int): Number of ECG leads.
        job: Job object (for executing commands).
        idExp: Experiment ID.
        plot (bool): If True, generates plots. If False, only saves data.
    """
    healthy_ECG = re.sub(r"vol\d+", "vol0", ECG)

    fibrotic_leads = []  # List to store fibrotic simulation leads
    healthy_leads = []   # List to store healthy simulation leads

    for i in range(num_leads):
        vtx_endo = str(i)
        vtx_epi = str(i + num_leads)

        # Fibrotic simulation data extraction
        endo_file = os.path.join(ECG, f"phie_elec1_{i}.dat")
        epi_file = os.path.join(ECG, f"phie_elec2_{i}.dat")

        extract_endo = [
            settings.execs.igbextract, "-l", vtx_endo, "-O", endo_file, "-o", "ascii",
            os.path.join(ECG, "phie_recovery.igb")
        ]
        job.bash(extract_endo)

        extract_epi = [
            settings.execs.igbextract, "-l", vtx_epi, "-O", epi_file, "-o", "ascii",
            os.path.join(ECG, "phie_recovery.igb")
        ]
        job.bash(extract_epi)

        endo_trace = txt.read(endo_file)
        epi_trace = txt.read(epi_file)
        lead_ecg = endo_trace - epi_trace
        fibrotic_leads.append(lead_ecg)

        # Healthy simulation data extraction
        healthy_endo_file = os.path.join(healthy_ECG, f"phie_elec1_{i}.dat")
        healthy_epi_file = os.path.join(healthy_ECG, f"phie_elec2_{i}.dat")

        healthy_extract_endo = [
            settings.execs.igbextract, "-l", vtx_endo, "-O", healthy_endo_file, "-o", "ascii",
            os.path.join(healthy_ECG, "phie_recovery.igb")
        ]
        job.bash(healthy_extract_endo)

        healthy_extract_epi = [
            settings.execs.igbextract, "-l", vtx_epi, "-O", healthy_epi_file, "-o", "ascii",
            os.path.join(healthy_ECG, "phie_recovery.igb")
        ]
        job.bash(healthy_extract_epi)

        healthy_endo_trace = txt.read(healthy_endo_file)
        healthy_epi_trace = txt.read(healthy_epi_file)
        healthy_lead_ecg = healthy_endo_trace - healthy_epi_trace
        healthy_leads.append(healthy_lead_ecg)

    # Convert lists to NumPy arrays and stack them column-wise
    fibrotic_leads_array = np.column_stack(fibrotic_leads)
    healthy_leads_array = np.column_stack(healthy_leads)

    # Save to CSV file
    fibrotic_csv_path = os.path.join(ECG, "fibrotic_ECG.csv")
    healthy_csv_path = os.path.join(healthy_ECG, "healthy_ECG.csv")

    np.savetxt(fibrotic_csv_path, fibrotic_leads_array, delimiter=",", fmt="%.6f")
    np.savetxt(healthy_csv_path, healthy_leads_array, delimiter=",", fmt="%.6f")

    print(f"Fibrotic ECG leads saved to: {fibrotic_csv_path}")
    print(f"Healthy ECG leads saved to: {healthy_csv_path}")

    # Skip plotting if plot=False
    if not plot:
        return

    # Plotting
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
