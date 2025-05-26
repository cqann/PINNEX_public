
import numpy as np
import torch
import os
import pandas as pd
# from src.networks.pinn_wrapper import CachedPINNWrapper # Assuming this is correctly defined elsewhere


def write_pinnex_predictions_streamed(model,
                                      block_pts_file='block.pts',
                                      ecg_parquet_file='ecg.parquet',
                                      sim_id=None,
                                      filename='predicted_Ts.dat',
                                      batch_size=1024,
                                      CV=False):
    # Ensure CachedPINNWrapper is defined or imported if you uncomment its usage
    # For this example, I'll assume model is a standard PyTorch model.
    # If CachedPINNWrapper is essential, ensure it's available.
    # from src.networks.pinn_wrapper import CachedPINNWrapper
    from src.networks.pinn_wrapper import CachedPINNWrapper

    L0 = 50000.0
    T0 = 215.0
    T0 = 810 if CV else T0
    print("Starting PINN-ECG prediction process...")

    # ----------------------------
    # 1) Read block.pts in order
    # ----------------------------
    print(f"Reading block.pts file: {block_pts_file}...")
    with open(block_pts_file, 'r') as f:
        lines = f.read().strip().split('\n')

    n_points = int(lines[0])
    coords = []
    for i in range(n_points):
        x_str, y_str, z_str = lines[i + 1].split()
        coord = [float(x_str), float(y_str), float(z_str)]
        coord = [k / L0 for k in coord]  # Normalize coordinates
        coords.append(coord)
    coords = np.array(coords, dtype=np.float32)
    print(f"Loaded {n_points} spatial points.")

    # ---------------------------------------
    # 2) Load ECG data
    # ---------------------------------------
    if sim_id is None:
        raise ValueError("sim_id must be provided.")
    print(f"Loading ECG data for sim_id {sim_id} from {ecg_parquet_file}...")
    df_ecg = pd.read_parquet(ecg_parquet_file)
    df_ecg_sim = df_ecg[df_ecg['sim_id'] == sim_id]
    if df_ecg_sim.empty:
        raise ValueError(f"No ECG data found for sim_id {sim_id}.")
    ecg_obj = df_ecg_sim.iloc[0]['ecg']

    if isinstance(ecg_obj, np.ndarray) and ecg_obj.dtype == np.object_:
        ecg_obj = np.array([np.array(lead, dtype=np.float32) for lead in ecg_obj])
    elif isinstance(ecg_obj, list):
        ecg_obj = np.array(ecg_obj, dtype=np.float32)
    ecg_tensor = torch.tensor(ecg_obj, dtype=torch.float32)
    print(f"Loaded ECG data with shape {ecg_tensor.shape}.")

    # -----------------------------------
    # 3) Prepare for inference
    # -----------------------------------
    device = next(model.parameters()).device
    model.eval()
    ecg_tensor = ecg_tensor.to(device)

    # Assuming CachedPINNWrapper is defined. If not, model itself should be callable.
    # For simplicity, if CachedPINNWrapper is not available, this example might need adjustment
    # on how 'wrapper' is defined or used.
    # wrapper = CachedPINNWrapper(model) # Uncomment if CachedPINNWrapper is available
    # wrapper.start_new_epoch()          # Uncomment if CachedPINNWrapper is available
    # Fallback if wrapper is not used:
    pinn_model = model  # Use the model directly if no wrapper
    wrapper = CachedPINNWrapper(model)
    wrapper.start_new_epoch()

    print("Streaming spatial data and running inference...")
    with torch.no_grad():
        with open(filename, 'w') as out_file:
            start_point = 0
            while start_point < n_points:
                end_point = min(start_point + batch_size, n_points)
                coords_batch = coords[start_point:end_point, :]
                x_batch = torch.tensor(coords_batch[:, 0], device=device).unsqueeze(1)
                y_batch = torch.tensor(coords_batch[:, 1], device=device).unsqueeze(1)
                z_batch = torch.tensor(coords_batch[:, 2], device=device).unsqueeze(1)

                current_batch_size = end_point - start_point
                ecg_batch = ecg_tensor.unsqueeze(0).expand(current_batch_size, -1, -1)
                sim_ids_batch = torch.full((current_batch_size,), sim_id, device=device)

                # Adjust this part based on your model's actual output structure
                # Assuming model directly returns T_pred or (T_pred, c_pred)
                # If using CachedPINNWrapper:
                # T_pred, c_pred = wrapper(x_batch, y_batch, z_batch, ecg_batch, sim_ids=sim_ids_batch)
                # If model is used directly and returns a tuple:
                output = wrapper(x_batch, y_batch, z_batch, ecg_batch,
                                 sim_ids_batch)  # Modify call as per your model
                if isinstance(output, tuple):  # Assuming (T_pred, c_pred) structure
                    T_pred, c_pred = output
                else:  # Assuming model directly returns T_pred
                    T_pred = output
                    c_pred = None  # or some default if CV logic relies on it

                T_pred = c_pred if CV and c_pred is not None else T_pred
                T_pred = T_pred * T0
                T_np = T_pred.cpu().numpy()

                for row in T_np:
                    out_file.write(f"{row[0]}\n")
                start_point = end_point
    print(f"Prediction process completed. Results written to {filename}")

    # ---------------------------------------------------
    # 5) Convert the .dat file to IGB format with a detailed header
    # ---------------------------------------------------
    igb_file_name = filename.replace('.dat', '.igb')

    # Define header parameters based on your reference IGB file (e.g., diff_sim_splash116.igb)
    # Adjust these values to match your specific reference header.
    ref_header_values = {
        "x_dim": n_points,  # This is determined by your block_pts_file
        "y_dim": 1,
        "z_dim": 1,
        # For the --create step, t_dim must be 1 because your .dat file contains n_points scalar values.
        "t_dim_for_create_step": 1,
        "data_type": "float",
        "system": "little",         # From "Created on: little_endian"
        "dim_x_size": "1.3856e-05",  # Example: X size from reference header
        "dim_y_size": "1",          # Example: Y size (if y_dim=1, often 1 unit)
        "dim_z_size": "1",          # Example: Z size (if z_dim=1, often 1 unit)
        "dim_t_size": "100",        # Example: T size (Duration) from reference header
        "inc_x": "1.91988e-10",     # Example: Increment in x from reference
        "inc_y": "1",               # Example: Increment in y (if y_dim=1, often 1)
        "inc_z": "1",               # Example: Increment in z (if z_dim=1, often 1)
        "inc_t": "1",               # Example: Increment in t from reference
        "data_factor": "1",         # Example: Pixel scaling / Data factor from reference
        "x_units": "um",            # Example: X units from reference
        "y_units": "um",            # Example: Y units from reference
        "z_units": "um",            # Example: Z units from reference
        "t_units": "ms",            # Example: T units (Pixel units) from reference
        # --- Optional fields from reference header ---
        # "author": "YourName",
        # "comment": "Predicted activation times",
        # "org_x": "0.0", "org_y": "0.0", "org_z": "0.0", "org_t": "0.0",
        # "data_zero": "0.0"
    }

    # Construct the igbhead command
    cmd_parts = [
        "igbhead",
        f"-x{ref_header_values['x_dim']}",
        f"-y{ref_header_values['y_dim']}",
        f"-z{ref_header_values['z_dim']}",
        f"-t{ref_header_values['t_dim_for_create_step']}",  # Crucially 1 for this step
        f"-d{ref_header_values['data_type']}",
        f"-s{ref_header_values['system']}",
        f"-X{ref_header_values['dim_x_size']}",
        f"-Y{ref_header_values['dim_y_size']}",
        f"-Z{ref_header_values['dim_z_size']}",
        f"-T{ref_header_values['dim_t_size']}",
        f"-I{ref_header_values['inc_x']}",
        f"-J{ref_header_values['inc_y']}",
        f"-K{ref_header_values['inc_z']}",
        f"-L{ref_header_values['inc_t']}",
        f"-S{ref_header_values['data_factor']}",
        f"-1\"{ref_header_values['x_units']}\"",  # Quote units if they can have spaces
        f"-2\"{ref_header_values['y_units']}\"",
        f"-3\"{ref_header_values['z_units']}\"",
        f"-4\"{ref_header_values['t_units']}\"",
    ]

    # Add optional fields if present in ref_header_values
    if "org_x" in ref_header_values:
        cmd_parts.append(f"-o{ref_header_values['org_x']}")
    if "org_y" in ref_header_values:
        cmd_parts.append(f"-p{ref_header_values['org_y']}")
    if "org_z" in ref_header_values:
        cmd_parts.append(f"-q{ref_header_values['org_z']}")
    if "org_t" in ref_header_values:
        cmd_parts.append(f"-r{ref_header_values['org_t']}")
    if "data_zero" in ref_header_values:
        cmd_parts.append(f"-0{ref_header_values['data_zero']}")
    if "author" in ref_header_values:
        cmd_parts.append(f"-A{ref_header_values['author']}")  # No spaces in author
    if "comment" in ref_header_values:
        cmd_parts.append(f"-c\"{ref_header_values['comment']}\"")

    cmd_parts.extend([
        "--create",
        f"-f{igb_file_name}",
        filename
    ])
    cmd = " ".join(cmd_parts)

    print(f"Creating IGB file with command: {cmd}")
    result = os.system(cmd)

    if result == 0:
        print(f"Successfully created IGB file: {igb_file_name}")
        os.remove(filename)
        print(f"Temporary prediction file {filename} removed.")

        # --- Scenario B (Optional & Use with Caution) ---
        # If you absolutely need the header to report a different t_dimension
        # than what the data supports (e.g., t=101 like diff_sim_splash116.igb)
        #
        # reference_t_dim_from_sim_file = 101 # The t-dim from your diff_sim_splash116.igb
        # if ref_header_values['t_dim_for_create_step'] != reference_t_dim_from_sim_file:
        #     print(f"\nWARNING: The created IGB file ({igb_file_name}) has t_dimension={ref_header_values['t_dim_for_create_step']} "
        #           f"to match the {n_points} data values.")
        #     print(f"Your reference file seems to have t_dimension={reference_t_dim_from_sim_file}.")
        #     print("If you force the header to this t_dimension, it will not match the actual data quantity.")
        #
        #     # Command to forcibly change ONLY the t-dimension in the header of the already created IGB file
        #     # cmd_force_t_dim = f"igbhead -t{reference_t_dim_from_sim_file} {igb_file_name}"
        #     # print(f"Executing (with caution): {cmd_force_t_dim}")
        #     # force_result = os.system(cmd_force_t_dim)
        #     # if force_result == 0:
        #     #     print(f"Header of {igb_file_name} modified to t_dimension={reference_t_dim_from_sim_file}.")
        #     # else:
        #     #     print(f"Failed to modify t_dimension for {igb_file_name}. Status: {force_result}")
        # else:
        #     print(f"The created IGB file's t_dimension ({ref_header_values['t_dim_for_create_step']}) matches the intended reference structure's t_dimension.")

    else:
        print(f"Error creating IGB file. igbhead command failed with status {result}.")
        print(f"The temporary .dat file {filename} has been kept for inspection.")

    print("Process complete.")

# Example usage (you'll need to define/load 'model' and have the input files):
# from your_model_module import YourPinnModel
# model = YourPinnModel(...)
# model.load_state_dict(torch.load('your_model.pth'))
#
# write_pinnex_predictions_streamed(
#     model=model,
#     block_pts_file='path/to/your/block.pts',
#     ecg_parquet_file='path/to/your/ecg.parquet',
#     sim_id=116, # Example sim_id
#     filename='predicted_activation_times.dat', # Temporary file
#     batch_size=1024
# )
