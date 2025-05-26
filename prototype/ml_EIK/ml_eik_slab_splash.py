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
        default=100.0,
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
    model = "ttpn"

    # ---------------------------------------
    # 1 : Define the mesh geometry
    # ---------------------------------------
    x = 20  # (mm)
    y = 20  # (mm)
    z = 2.5   # (mm)
    res = 0.25

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
    setup_fibrois(meshname, args, (x, y, z), 3500)  # 3500=1patch 2600=splash

    # Now, query the updated element tags for conduction
    _, new_etags, _ = txt.read(meshname + ".elem")
    all_tags = np.unique(new_etags)
    IntraTags = all_tags[all_tags != 0]  # for intracellular
    ExtraTags = all_tags.copy()          # for extracellular

    # ---------------------------------------
    # 2 : Define ionic models & conductivities
    # ---------------------------------------
    imp_reg = setup_ionic(args, model)
    g_reg = setup_gregions(args)

    # ---------------------------------------
    # 3 : Define the stimulation
    # ---------------------------------------
    stim = [
        "-num_stim", 1,
        "-stim[0].crct.type", 0,
        "-stim[0].pulse.strength", 2000.0,
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

    if model == "AlievPanfilov":
        cmd += [
            "-num_gvecs", 1,
            "-gvec[0].name", "V",
            "-gvec[0].units", "mV",
            "-gvec[0].imp", "AlievPanfilov",
            "-gvec[0].ID[0]", "V",
            "-gvec[0].bogus", 0
        ]

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


def setup_ionic(args, model="ttpn"):
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

    n_regions = 3 if args.fibrosis_vol_fraction > 0 or args.n_patches > 0 else 1

    imp_reg = ["-num_imp_regions", n_regions]
    imp_names = ["healthy", "fibrotic", "non-conductive"]

    for i in range(n_regions):
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


def setup_gregions(args):
    """
    Healthy conduction: gi=0.174, ge=0.625
    Fibrotic conduction changes:
      - 80% reduction in transverse => g_it = 0.2 * gi
      - 50% reduction in longitudinal => g_il = 0.5 * gi
      => anisotropy ratio ~ 2.5x
    Region 6 = non-conductive => set conduction to 0.
    """
    CV_healthy = 500  # (mm/ms)
    CV_ar = 1 / 0.42
    CV_ar = 1

    g_reg = [
        "-num_gregions", (3 if args.fibrosis_vol_fraction > 0 or args.n_patches > 0 else 1),

        # region 0 => tag=1, healthy
        "-gregion[0].num_IDs", 1,
        "-gregion[0].ID[0]", 1,
        "-gregion[0].CV.vel", CV_healthy,
        "-gregion[0].CV.ar_t", CV_ar,
        "-gregion[0].CV.ar_n", CV_ar,
    ]

    if args.fibrosis_vol_fraction == 0 and args.n_patches == 0:
        return g_reg

    g_reg += [
        # region 2 => tag=3, fibrotic myocytes
        "-gregion[1].num_IDs", 1,
        "-gregion[1].ID[0]", 2,
        "-gregion[1].CV.vel", CV_healthy * 0.25,
        "-gregion[1].CV.ar_t", CV_ar,
        "-gregion[1].CV.ar_n", CV_ar,


        # region 5 => tag=6, non-conductive ECM
        "-gregion[2].num_IDs", 1,
        "-gregion[2].ID[0]", 3,
        "-gregion[2].CV.vel", CV_healthy * 0.25,
        "-gregion[2].CV.ar_t", CV_ar,
        "-gregion[2].CV.ar_n", CV_ar,
    ]
    return g_reg


def setup_fibrois(meshname, args, mesh_sz, dot_size):
    # Define fibrotic regions
    x, y, z = mesh_sz
    elems, etags, nelems = txt.read(meshname + ".elem")
    splash = args.n_patches == 0 and args.fibrosis_vol_fraction > 0

    fibrotic_points = []
    fibrotic_tetra_set = set()
    vertex_to_tetra = {}
    for i, tet in enumerate(elems):
        for vertex in tet:
            vertex_to_tetra.setdefault(vertex, []).append(i)

    n_tissue_elems = len([x for x in etags if x != 0])
    fib_regs = 0

    random.seed(args.seed)
    while True:
        if not splash and fib_regs >= args.n_patches:
            break
        elif splash and len(fibrotic_tetra_set) > args.fibrosis_vol_fraction * n_tissue_elems:
            break
        fibrotic_seed_x = random.uniform(0, 1)
        fibrotic_seed_y = random.uniform(0, 1)
        fibrotic_seed_z = random.uniform(0, 1)

        if (fibrotic_seed_x ** 2 + (fibrotic_seed_y - 0.5) ** 2) <= 0.15:
            continue

        fibrotic_seed_x = fibrotic_seed_x * x * 1000
        fibrotic_seed_y = fibrotic_seed_y * y * 1000
        fibrotic_seed_z = fibrotic_seed_z * z * 1000

        mean_target = dot_size  # Expected value
        low, high = mean_target * 0.5, mean_target * 2  # Typical range
        sigma = math.log(high / mean_target) / 2  # Rough estimation
        mu = math.log(mean_target) - (sigma ** 2) / 2
        thr_radius = random.lognormvariate(mu, sigma)
        thr_radius = random.uniform(0.7 * dot_size, 1.3 * dot_size)  # random.uniform(0.5 * dot_size, 1.5 * dot_size)
        command = ["meshtool", "query", "idx", "-msh=" + meshname,
                   f"-coord={fibrotic_seed_x},{fibrotic_seed_y},{fibrotic_seed_z}", f"-thr={thr_radius}"]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            str_split = result.stdout.split("Vertex list:")
            if len(str_split) != 2:
                fibrotic_points = []
            else:
                fibrotic_points = [int(x) for x in str_split[1].strip().split(",")]

        except subprocess.CalledProcessError as e:
            print("Error executing meshtool command:")
            print(e.stderr)
            return

        fibrotic_tetra = set()
        for vertex in fibrotic_points:
            fibrotic_tetra.update(vertex_to_tetra.get(vertex, []))
        fibrotic_tetra = list(fibrotic_tetra)
        fibrotic_tetra_set = fibrotic_tetra_set.union(set(fibrotic_tetra))
        fib_regs += 1

    fibrotic_tetras = [i for i in fibrotic_tetra_set if etags[i] != 0]
    random.shuffle(fibrotic_tetras)
    etags[fibrotic_tetras] = 2

    # Write the updated data
    # IMPORTANT: 'elems' must remain shape (N,4) for Tt or (N,8) for Hx, etc.
    # 'etags=etags' is a keyword argument.

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


def igb_reader(igb_filename):
    """
    Reads an IGB file into NumPy arrays using openCARP command-line
    utilities igbhead and igbextract. Uses -o asciiTm so each line
    contains: [time, val_node0, val_node1, ..., val_node{n-1}].
    """

    # 1) Gather header metadata -----------------------------------------
    cmd_head = ["igbhead", igb_filename]
    try:
        result = subprocess.run(cmd_head, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running igbhead on {igb_filename}:\n{e.stderr}")

    header_output = result.stdout.splitlines()

    xdim = ydim = zdim = 1
    tdim = 1
    dt = None
    t_origin = 0.0

    for line in header_output:
        line_low = line.strip().lower()
        if line_low.startswith("x dimension:"):
            xdim = int(line.split(":")[1])
        elif line_low.startswith("y dimension:"):
            ydim = int(line.split(":")[1])
        elif line_low.startswith("z dimension:"):
            zdim = int(line.split(":")[1])
        elif line_low.startswith("t dimension:"):
            tdim = int(line.split(":")[1])
        elif line_low.startswith("increment in t:"):
            dt_str = line.split(":")[1].strip()
            dt = float(dt_str)
        elif line_low.startswith("t origin:"):
            t_origin_str = line.split(":")[1].strip()
            t_origin = float(t_origin_str)

    n_nodes = xdim * ydim * zdim
    if dt is None:
        dt = 1.0

    # 2) Use -o asciiTm for extraction: --------------------------------
    #
    #    Each line => time value + n_nodes data values
    #
    #    default example line =>  "time val_node0 val_node1 ... val_node_{n_nodes-1}"
    #
    cmd_extract = [
        "igbextract",
        f"-l0-{n_nodes - 1}",    # Extract from node 0 to node n_nodes-1
        "-O", "-",
        "-o", "asciiTm",        # Output format is asciiTm
        igb_filename
    ]

    try:
        result_extract = subprocess.run(cmd_extract, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running igbextract on {igb_filename}:\n{e.stderr}")

    extract_lines = result_extract.stdout.strip().split("\n")
    num_timesteps = len(extract_lines)
    if num_timesteps != tdim:
        print(f"Warning: Extracted {num_timesteps} lines, but header says {tdim} timesteps.")

    # 3) Parse lines into time array & data -----------------------------
    #
    #    For asciiTm:
    #      columns = [time, nodeVal0, nodeVal1, ..., nodeVal{n_nodes-1}]
    #
    time_array = np.zeros(num_timesteps, dtype=float)
    data = np.zeros((num_timesteps, n_nodes), dtype=float)

    for i, line in enumerate(extract_lines):
        cols = line.strip().split()
        if len(cols) != (1 + n_nodes):
            raise ValueError(
                f"Line {i} has {len(cols)} columns, expected {1 + n_nodes} "
                f"(time + {n_nodes} node vals)."
            )
        time_array[i] = float(cols[0])
        data_vals = [float(x) for x in cols[1:]]
        data[i, :] = data_vals

    # If you prefer to override the time array from dt + t_origin:
    # time_array = t_origin + np.arange(num_timesteps) * dt

    header_info = {
        "x_dim": xdim,
        "y_dim": ydim,
        "z_dim": zdim,
        "t_dim": tdim,
        "dt": dt,
        "t_origin": t_origin,
        "n_nodes": n_nodes,
    }

    return time_array, data, header_info


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


if __name__ == "__main__":
    run()
