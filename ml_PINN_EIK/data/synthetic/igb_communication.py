import subprocess
import numpy as np


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


def example_usage():
    igb_file = "vm.igb"
    time_array, vm_data, hdr = igb_reader(igb_file)

    print("\n--- IGB Reader Results ---")
    print(f"Header info: {hdr}")
    print(f"Time array shape: {time_array.shape}")
    print(f"Data array shape: {vm_data.shape}")
    print(f"First few times: {time_array[:5]}")
    print(f"First row of data (time= {time_array[0]}): {vm_data[0, :5]}")


if __name__ == "__main__":
    example_usage()
