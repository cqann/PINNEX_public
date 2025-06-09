import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
import torch  # Not used in this snippet directly, but kept for context
from scipy.stats import pearsonr

# --- Path Setup for Imports ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()

project_root_guess = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
if not os.path.isdir(os.path.join(project_root_guess, 'src')):
    project_root_guess = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    if not os.path.isdir(os.path.join(project_root_guess, 'src')):
        project_root_guess = current_script_dir

src_path = os.path.join(project_root_guess, 'src')
evaluation_module_path = os.path.join(project_root_guess, 'notebooks', 'evaluation')

if src_path not in sys.path:
    sys.path.append(src_path)
if evaluation_module_path not in sys.path:
    sys.path.append(evaluation_module_path)

try:
    from utils.igb_utils import write_pinnex_predictions_streamed
    from igb_communication import igb_reader
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import necessary modules (write_pinnex_predictions_streamed or igb_reader): {e}")
    print("Ensure 'src/utils/igb_utils.py' and 'notebooks/evaluation/igb_communications.py' are accessible.")
    sys.exit(1)


def cosine_similarity_1d(vec1, vec2):
    vec1, vec2 = np.asarray(vec1).flatten(), np.asarray(vec2).flatten()
    if vec1.shape != vec2.shape:
        return np.nan
    norm_vec1, norm_vec2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2) if norm_vec1 > 0 and norm_vec2 > 0 else 0.0


# Changed from IGBOPS_EXEC to IGBHEAD_EXEC based on logs
IGBHEAD_EXEC = "igbhead"
IGBOPS_EXEC = "igbops"  # Still needed for X-Y expressions


def run_subprocess_command(command_list):
    try:
        # print(f"DEBUG CMD: {' '.join(command_list)}") # Optional: for debugging the command
        result = subprocess.run(command_list, check=True, capture_output=True, text=True, errors='ignore')
        # print(f"CMD OK: {' '.join(command_list)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"CMD FAILED: {' '.join(e.cmd)}\nExit {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"CMD FAILED: Executable not found: {command_list[0]}")
        return False


def calculate_igb_difference(input_x_igb, input_y_igb, output_diff_igb):
    if not (os.path.exists(input_x_igb) and os.path.exists(input_y_igb)):
        print(
            f"Error: One or both input IGB files for difference calculation do not exist: {input_x_igb}, {input_y_igb}")
        return False
    cmd = [IGBOPS_EXEC, "--expr", "X-Y", input_x_igb, input_y_igb, "-O", output_diff_igb]
    return run_subprocess_command(cmd)


def _load_activation_data(igb_file_path):
    """Helper to load a single activation map from an IGB file."""
    try:
        _, raw_data, header = igb_reader(igb_file_path)
        if raw_data is not None and raw_data.shape[0] > 0 and raw_data.shape[1] > 0:
            return raw_data[0, :]
        print(
            f"Warning: No valid data found in {os.path.basename(igb_file_path)} (t_dim={header.get('t_dim', 'N/A')}, n_nodes={header.get('n_nodes', 'N/A')}).")
    except Exception as e:
        print(f"Error reading IGB {os.path.basename(igb_file_path)}: {e}")
    return None

# --- UPDATED HELPER FUNCTION to create an IGB file with a constant value using igbhead ---


def _create_igb_from_constant_value(
    value,
    output_igb_path,
    ref_header,  # Reference header dictionary from a similar IGB file
    force_regenerate=False
):
    """
    Creates an IGB file where all nodes have a constant activation value using igbhead.
    It attempts to mirror the igbhead command structure observed in logs.
    """
    if not force_regenerate and os.path.exists(output_igb_path):
        return True

    # Use a more distinguishable name for the temp .dat file to avoid potential collisions
    # if `value` is part of the output_igb_path already.
    base_name_for_dat = os.path.splitext(os.path.basename(output_igb_path))[0]
    temp_dat_path = os.path.join(os.path.dirname(output_igb_path), f"{base_name_for_dat}_const_val_{value:.4f}.dat")

    try:
        num_nodes = ref_header.get('n_nodes')
        # t_dim for activation maps is typically 1. Get from ref_header if available, else default.
        t_dim = ref_header.get('t_dim', 1)
        # t_step for -T parameter of igbhead. From example log, this was 100.
        # ref_header['t_step'] or ref_header['dt'] should provide this.
        # Defaulting to 100.0 from example if not in header.
        t_step_for_T_param = ref_header.get('t_step', ref_header.get('dt', 100.0))

        if num_nodes is None:
            print(f"  ERROR: 'n_nodes' not found in reference header for generating {output_igb_path}")
            return False

        constant_data = np.full((t_dim, int(num_nodes)), value, dtype=np.float32)  # Ensure num_nodes is int for np.full
        constant_data.tofile(temp_dat_path)

        # Construct igbhead command based on ref_header and example log.
        # Parameters that seem static from the log are hardcoded.
        # Others (num_nodes, t_dim, t_step_for_T_param) are from ref_header.
        cmd = [
            IGBHEAD_EXEC,
            f"-x{int(num_nodes)}",
            "-y1",  # From example log
            "-z1",  # From example log
            f"-t{int(t_dim)}",
            "-dfloat",
            "-slittle",  # From example log (system endianness)

            # Spacing/Scaling factors and other parameters from example log
            "-X1.3856e-05",
            "-Y1",
            "-Z1",
            f"-T{t_step_for_T_param}",  # Use t_step from ref_header, should be 100 based on example
            "-I1.91988e-10",
            "-J1",
            "-K1",
            "-L1",
            "-S1",

            # Units from example log
            "-1\"um\"",
            "-2\"um\"",
            "-3\"um\"",
            "-4\"ms\"",

            "--create",
            f"-f{output_igb_path}",  # Output IGB file path
            temp_dat_path           # Input DAT file path
        ]

        # print(f"  DEBUG: igbhead command: {' '.join(cmd)}") # For debugging

        if not run_subprocess_command(cmd):
            print(f"  ERROR: igbhead failed for {output_igb_path}.")
            if os.path.exists(temp_dat_path):
                try:
                    os.remove(temp_dat_path)
                except OSError as e_rm:
                    print(f"  Warning: Could not remove temp .dat file {temp_dat_path}: {e_rm}")
            return False

        if os.path.exists(temp_dat_path):
            try:
                os.remove(temp_dat_path)
            except OSError as e_rm:
                print(f"  Warning: Could not remove temp .dat file {temp_dat_path} post-conversion: {e_rm}")

        return os.path.exists(output_igb_path)

    except Exception as e:
        print(f"  ERROR in _create_igb_from_constant_value for {output_igb_path}: {e}")
        if os.path.exists(temp_dat_path):
            try:
                os.remove(temp_dat_path)
            except OSError as e_rm:
                print(f"  Warning: Could not remove temp .dat file {temp_dat_path} on error: {e_rm}")
        return False


def evaluate_fibrosis_detection(
    model,
    healthy_parquet_sim_id,
    ecg_parquet_file_abs_path,
    block_pts_file_abs_path,  # May not be directly used by igbhead if all info in ref_header
    delay_file_abs_path,
    sim_raw_base_abs_path,
    model_name_tag="pinn_model",
    CV_predictions=False,
    prediction_batch_size=1024,
    force_regenerate_predictions=False,
    force_regenerate_diffs=False
):
    metrics_diff_pcc = []
    metrics_diff_cosine = []
    metrics_direct_fib_pcc = []
    metrics_direct_fib_cosine = []

    healthy_parquet_sim_id = int(healthy_parquet_sim_id)
    print(f"--- Evaluation: {model_name_tag} | Healthy Parquet ID: {healthy_parquet_sim_id} ---")

    if not all(os.path.exists(p) for p in [ecg_parquet_file_abs_path, block_pts_file_abs_path, sim_raw_base_abs_path, delay_file_abs_path]):
        print("ERROR: Essential input file/directory (ECG parquet, block_pts, delay_file, or sim_raw base) not found. Aborting.")
        return None

    try:
        df_ecg = pd.read_parquet(ecg_parquet_file_abs_path, columns=['sim_id'])
        all_parquet_sim_ids = df_ecg['sim_id'].unique().tolist()
        if healthy_parquet_sim_id not in all_parquet_sim_ids:
            print(f"ERROR: Healthy Parquet ID {healthy_parquet_sim_id} not in Parquet. Aborting.")
            return None
    except Exception as e:
        print(f"ERROR reading Parquet {ecg_parquet_file_abs_path}: {e}. Aborting.")
        return None

    try:
        with open(delay_file_abs_path, 'r') as delay_file:
            delay_data_lines = delay_file.readlines()
        delay_map = {}
        for line in delay_data_lines:
            parquet_id, delay_val_str = line.strip().split(',')
            delay_map[int(parquet_id)] = float(delay_val_str)
        print(f"Loaded delay file with {len(delay_map)} entries.")
    except Exception as e:
        print(f"ERROR reading or parsing delay file {delay_file_abs_path}: {e}. Aborting.")
        return None

    delay_H_reference = delay_map.get(healthy_parquet_sim_id)
    if delay_H_reference is None:
        print(
            f"WARNING: Delay for healthy reference Parquet ID {healthy_parquet_sim_id} not found in delay map. Assuming 0.0 ms.")
        delay_H_reference = 0.0

    healthy_sim_raw_dir_id_str = str(healthy_parquet_sim_id // 10)
    healthy_sim_output_dir = os.path.join(sim_raw_base_abs_path, healthy_sim_raw_dir_id_str)
    os.makedirs(healthy_sim_output_dir, exist_ok=True)
    sim_healthy_act_path = os.path.join(healthy_sim_output_dir, "act.igb")
    if not os.path.exists(sim_healthy_act_path):
        print(f"ERROR: Ground truth healthy act.igb not found: {sim_healthy_act_path}. Aborting.")
        return None

    model_pred_healthy_igb_path = os.path.join(
        healthy_sim_output_dir, f"model_pred_healthy_parquetID_{healthy_parquet_sim_id}_{model_name_tag}.igb")
    if force_regenerate_predictions or not os.path.exists(model_pred_healthy_igb_path):
        print(f"Predicting HEALTHY: Parquet ID {healthy_parquet_sim_id}")
        temp_dat_path = model_pred_healthy_igb_path.replace(".igb", ".dat")
        # write_pinnex_predictions_streamed is assumed to use igbhead internally as per logs
        write_pinnex_predictions_streamed(
            model, block_pts_file_abs_path, ecg_parquet_file_abs_path, healthy_parquet_sim_id,
            temp_dat_path, prediction_batch_size, CV_predictions
        )
        # Check if the IGB file was created by write_pinnex_predictions_streamed at the expected location
        # (write_pinnex_predictions_streamed output path for IGB is based on its internal logic, often related to temp_dat_path)
        # The provided logs show write_pinnex_predictions_streamed creates the IGB file directly with -f<output_igb_file>
        # where output_igb_file is the path passed implicitly (derived from temp_dat_path).
        # So, model_pred_healthy_igb_path should exist IF write_pinnex_predictions_streamed uses that convention.
        # The existing shutil.move logic might be redundant if write_pinnex_predictions_streamed now correctly
        # places the IGB file. For safety, we keep it simple: check if the target exists.
        if not os.path.exists(model_pred_healthy_igb_path):
            # Fallback: if write_pinnex_predictions_streamed created it as temp_dat_path + .igb
            expected_igb_from_func = temp_dat_path.replace(".dat", ".igb")
            if os.path.exists(expected_igb_from_func):
                if expected_igb_from_func != model_pred_healthy_igb_path:  # Ensure correct final name
                    shutil.move(expected_igb_from_func, model_pred_healthy_igb_path)
            else:
                print(
                    f"ERROR: Failed to generate model healthy prediction: {model_pred_healthy_igb_path} (or its .dat counterpart's .igb). Aborting.")
                return None

        if not os.path.exists(model_pred_healthy_igb_path):  # Final check
            print(
                f"ERROR: Model healthy prediction IGB not found after attempt: {model_pred_healthy_igb_path}. Aborting.")
            return None

    try:
        _, _, ref_header_dict = igb_reader(model_pred_healthy_igb_path)
        if not ref_header_dict or 'n_nodes' not in ref_header_dict:
            print(
                f"ERROR: Could not read header or n_nodes from reference IGB {model_pred_healthy_igb_path}. Aborting.")
            return None
        # No need to extract individual params here anymore if _create_igb_from_constant_value handles them from ref_header_dict
    except Exception as e:
        print(f"ERROR reading reference header from {model_pred_healthy_igb_path}: {e}. Aborting.")
        return None

    fibrotic_parquet_sim_ids = [pid for pid in all_parquet_sim_ids if pid // 10 != healthy_parquet_sim_id // 10]
    if not fibrotic_parquet_sim_ids:
        print("No fibrotic simulations to process.")
        return {}

    cached_simulated_diff_paths = {}

    for fib_parquet_sim_id in fibrotic_parquet_sim_ids:
        print(f"\nProcessing FIBROTIC: Parquet ID {fib_parquet_sim_id}")
        fib_sim_raw_dir_id_str = str(fib_parquet_sim_id // 10)
        current_sim_proc_dir = os.path.join(sim_raw_base_abs_path, fib_sim_raw_dir_id_str)
        os.makedirs(current_sim_proc_dir, exist_ok=True)

        sim_fibrotic_act_path = os.path.join(current_sim_proc_dir, "act.igb")
        if not os.path.exists(sim_fibrotic_act_path):
            print(f"  Skipping: Ground truth fibrotic act.igb not found: {sim_fibrotic_act_path}")
            continue

        sim_diff_cache_key = (fib_sim_raw_dir_id_str, healthy_sim_raw_dir_id_str)
        diff_simulated_igb_path = cached_simulated_diff_paths.get(sim_diff_cache_key)

        diff_sim_filename = f"diff_simulated_base{fib_sim_raw_dir_id_str}_vs_base{healthy_sim_raw_dir_id_str}.igb"
        potential_diff_simulated_igb_path = os.path.join(current_sim_proc_dir, diff_sim_filename)

        if diff_simulated_igb_path is None:
            diff_simulated_igb_path = potential_diff_simulated_igb_path

        if force_regenerate_diffs or not os.path.exists(diff_simulated_igb_path):
            print(f"  Generating simulated difference: {os.path.basename(diff_simulated_igb_path)}")
            if not calculate_igb_difference(sim_fibrotic_act_path, sim_healthy_act_path, diff_simulated_igb_path):
                print(f"  Skipping: Failed to create simulated difference for base {fib_sim_raw_dir_id_str}")
                continue
            cached_simulated_diff_paths[sim_diff_cache_key] = diff_simulated_igb_path

        model_pred_fibrotic_igb_path = os.path.join(
            current_sim_proc_dir, f"model_pred_fibrotic_parquetID_{fib_parquet_sim_id}_{model_name_tag}.igb")
        if force_regenerate_predictions or not os.path.exists(model_pred_fibrotic_igb_path):
            print(f"  Predicting FIBROTIC: Parquet ID {fib_parquet_sim_id}")
            temp_dat_path = model_pred_fibrotic_igb_path.replace(".igb", ".dat")
            write_pinnex_predictions_streamed(
                model, block_pts_file_abs_path, ecg_parquet_file_abs_path, fib_parquet_sim_id,
                temp_dat_path, prediction_batch_size, CV_predictions
            )
            if not os.path.exists(model_pred_fibrotic_igb_path):
                expected_igb_from_func = temp_dat_path.replace(".dat", ".igb")
                if os.path.exists(expected_igb_from_func) and expected_igb_from_func != model_pred_fibrotic_igb_path:
                    shutil.move(expected_igb_from_func, model_pred_fibrotic_igb_path)

            if not os.path.exists(model_pred_fibrotic_igb_path):
                print(f"  Skipping: Failed to generate model fibrotic prediction for Parquet ID {fib_parquet_sim_id}")
                continue

        diff_model_pred_igb_path = os.path.join(
            current_sim_proc_dir, f"diff_model_pred_ADJUSTED_parquetID_{fib_parquet_sim_id}_vs_healthy_{healthy_parquet_sim_id}_{model_name_tag}.igb")

        if force_regenerate_diffs or not os.path.exists(diff_model_pred_igb_path):
            print(f"  Generating DELAY-ADJUSTED model-predicted difference for Parquet ID {fib_parquet_sim_id}")

            delay_F_current = delay_map.get(fib_parquet_sim_id)
            if delay_F_current is None:
                print(
                    f"  WARNING: Delay for fibrotic Parquet ID {fib_parquet_sim_id} not found. Assuming 0.0 ms for this sim's D_F.")
                delay_F_current = 0.0

            relative_delay_value = delay_F_current - delay_H_reference
            print(
                f"    Healthy_Delay (D_H): {delay_H_reference:.2f} ms, Fibrotic_Delay (D_F): {delay_F_current:.2f} ms, Relative_Delay (D_F - D_H): {relative_delay_value:.2f} ms")

            if abs(relative_delay_value) < 1e-6:
                print("    Relative delay is near zero. Calculating model diff directly (Pred_Fib - Pred_Healthy).")
                if not calculate_igb_difference(model_pred_fibrotic_igb_path, model_pred_healthy_igb_path, diff_model_pred_igb_path):
                    print(
                        f"  Skipping: Failed to create model-predicted difference (zero relative delay) for Parquet ID {fib_parquet_sim_id}")
                    continue
            else:
                relative_delay_igb_path = os.path.join(
                    current_sim_proc_dir, f"const_relative_delay_val_{relative_delay_value:.2f}_{fib_parquet_sim_id}_{model_name_tag}.igb")

                if not _create_igb_from_constant_value(
                    relative_delay_value,
                    relative_delay_igb_path,
                    ref_header_dict,  # Pass the full reference header
                    force_regenerate=force_regenerate_diffs
                ):
                    print(f"  Skipping: Failed to create IGB file for relative delay {relative_delay_value:.2f} ms.")
                    continue

                temp_diff_unadjusted_igb_path = os.path.join(
                    current_sim_proc_dir, f"temp_PFminusPH_diff_{fib_parquet_sim_id}_{model_name_tag}.igb")
                if not calculate_igb_difference(model_pred_fibrotic_igb_path, model_pred_healthy_igb_path, temp_diff_unadjusted_igb_path):
                    print(
                        f"  Skipping: Failed to calculate (Pred_Fib - Pred_Healthy) for Parquet ID {fib_parquet_sim_id}")
                    if os.path.exists(relative_delay_igb_path):
                        os.remove(relative_delay_igb_path)
                    continue

                if not calculate_igb_difference(temp_diff_unadjusted_igb_path, relative_delay_igb_path, diff_model_pred_igb_path):
                    print(
                        f"  Skipping: Failed to subtract relative delay from model difference for Parquet ID {fib_parquet_sim_id}")
                    if os.path.exists(relative_delay_igb_path):
                        os.remove(relative_delay_igb_path)
                    if os.path.exists(temp_diff_unadjusted_igb_path):
                        os.remove(temp_diff_unadjusted_igb_path)
                    continue

                if os.path.exists(relative_delay_igb_path):
                    os.remove(relative_delay_igb_path)
                if os.path.exists(temp_diff_unadjusted_igb_path):
                    os.remove(temp_diff_unadjusted_igb_path)

        data_sim_diff = _load_activation_data(diff_simulated_igb_path)
        data_model_diff = _load_activation_data(diff_model_pred_igb_path)

        if data_sim_diff is None or data_model_diff is None:
            print(
                f"  Skipping diff metrics for Parquet ID {fib_parquet_sim_id} due to loading error for difference files (simulated or model-predicted).")
            metrics_diff_pcc.append(np.nan)  # Ensure list lengths match
            metrics_diff_cosine.append(np.nan)
        elif data_sim_diff.shape != data_model_diff.shape:
            print(
                f"  Skipping diff metrics for Parquet ID {fib_parquet_sim_id} due to shape mismatch in difference files (Sim: {data_sim_diff.shape}, Model: {data_model_diff.shape}).")
            metrics_diff_pcc.append(np.nan)
            metrics_diff_cosine.append(np.nan)
        else:
            data_sim_diff_mean = np.mean(data_sim_diff)
            data_sim_indices = np.where(data_sim_diff > data_sim_diff_mean)[0]
            data_model_diff_mean = np.mean(data_model_diff)
            data_model_indices = np.where(data_model_diff > data_model_diff_mean)[0]
            union_indices = np.union1d(data_sim_indices, data_model_indices)

            if len(union_indices) > 1:
                filtered_sim_diff = data_sim_diff[union_indices]
                filtered_model_diff = data_model_diff[union_indices]
                filtered_sim_diff = filtered_sim_diff - np.mean(filtered_sim_diff)
                filtered_model_diff = filtered_model_diff - np.mean(filtered_model_diff)

                if filtered_sim_diff.shape == filtered_model_diff.shape and len(filtered_sim_diff) > 1:
                    pcc, _ = pearsonr(filtered_sim_diff, filtered_model_diff)
                    cos_sim_diff = cosine_similarity_1d(filtered_sim_diff, filtered_model_diff)
                    metrics_diff_pcc.append(pcc)
                    metrics_diff_cosine.append(cos_sim_diff)
                    print(
                        f"  Diff Metrics (Adjusted) - PCC: {pcc:.4f}, CosineSim(Diff): {cos_sim_diff:.4f} (on {len(union_indices)} nodes)")
                else:
                    print(
                        f"  Skipping diff metrics for Parquet ID {fib_parquet_sim_id}: Not enough data points after filtering or shape mismatch.")
                    metrics_diff_pcc.append(np.nan)
                    metrics_diff_cosine.append(np.nan)
            else:
                print(
                    f"  Skipping diff metrics for Parquet ID {fib_parquet_sim_id}: Not enough data points in union_indices ({len(union_indices)}).")
                metrics_diff_pcc.append(np.nan)
                metrics_diff_cosine.append(np.nan)

        data_sim_fibrotic_raw = _load_activation_data(sim_fibrotic_act_path)
        data_model_pred_fibrotic_raw = _load_activation_data(model_pred_fibrotic_igb_path)

        if data_sim_fibrotic_raw is None or data_model_pred_fibrotic_raw is None:
            print(
                f"  Skipping direct metrics for Parquet ID {fib_parquet_sim_id} due to loading error for raw/predicted fibrotic files.")
            metrics_direct_fib_pcc.append(np.nan)  # Ensure list lengths match
            metrics_direct_fib_cosine.append(np.nan)
        elif data_sim_fibrotic_raw.shape != data_model_pred_fibrotic_raw.shape:
            print(
                f"  Skipping direct metrics for Parquet ID {fib_parquet_sim_id} due to shape mismatch (Sim: {data_sim_fibrotic_raw.shape}, Model: {data_model_pred_fibrotic_raw.shape}).")
            metrics_direct_fib_pcc.append(np.nan)
            metrics_direct_fib_cosine.append(np.nan)
        else:
            if len(data_sim_fibrotic_raw) > 1:
                pcc_fib, _ = pearsonr(data_sim_fibrotic_raw, data_model_pred_fibrotic_raw)
                metrics_direct_fib_pcc.append(pcc_fib)
                print(f"  Direct Fibrotic PCC: {pcc_fib:.4f}")
            else:
                metrics_direct_fib_pcc.append(np.nan)
                print(f"  Direct Fibrotic PCC: NaN (not enough data points: {len(data_sim_fibrotic_raw)})")
            cos_sim_direct_fib = cosine_similarity_1d(data_sim_fibrotic_raw, data_model_pred_fibrotic_raw)
            metrics_direct_fib_cosine.append(cos_sim_direct_fib)
            print(f"  Direct Fibrotic CosineSim: {cos_sim_direct_fib:.4f}")

        valid_diff_metrics = sum(1 for x in metrics_diff_pcc if not np.isnan(x))
        print(f"Progress: {valid_diff_metrics}/{len(fibrotic_parquet_sim_ids)} fibrotic cases processed for diff metrics.")

    results = {
        "avg_diff_pcc": np.nanmean(metrics_diff_pcc) if metrics_diff_pcc else np.nan,
        "avg_diff_cosine": np.nanmean(metrics_diff_cosine) if metrics_diff_cosine else np.nan,
        "avg_direct_fib_pcc": np.nanmean(metrics_direct_fib_pcc) if metrics_direct_fib_pcc else np.nan,
        "avg_direct_fib_cosine": np.nanmean(metrics_direct_fib_cosine) if metrics_direct_fib_cosine else np.nan,
        "individual_diff_pcc": metrics_diff_pcc,
        "individual_diff_cosine": metrics_diff_cosine,
        "individual_direct_fib_pcc": metrics_direct_fib_pcc,
        "individual_direct_fib_cosine": metrics_direct_fib_cosine,
        "num_fibrotic_cases_attempted": len(fibrotic_parquet_sim_ids),
        "num_fibrotic_cases_valid_diff_metrics": sum(1 for x in metrics_diff_pcc if not np.isnan(x))
    }
    print("\n--- Averaged Results (Delay Adjusted Model Difference) ---")
    for key, val in results.items():
        if "avg_" in key:
            print(f"{key}: {val:.4f}")
        elif "num_" in key:
            print(f"{key}: {val}")
    print("--- Evaluation Complete ---")
    return results
