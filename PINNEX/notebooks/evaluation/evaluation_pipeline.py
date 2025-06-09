
import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# --- Path Setup for Imports ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()

# Assuming standard project structure: project_root/src and project_root/notebooks/evaluation
project_root_guess = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
if not os.path.isdir(os.path.join(project_root_guess, 'src')):  # If script is already in project root or notebooks
    project_root_guess = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    if not os.path.isdir(os.path.join(project_root_guess, 'src')):
        project_root_guess = current_script_dir  # Guess project root is current dir

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
    sys.exit(1)  # Exit if critical dependencies are missing


def cosine_similarity_1d(vec1, vec2):
    vec1, vec2 = np.asarray(vec1).flatten(), np.asarray(vec2).flatten()
    if vec1.shape != vec2.shape:
        return np.nan
    norm_vec1, norm_vec2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2) if norm_vec1 > 0 and norm_vec2 > 0 else 0.0


IGBOPS_EXEC = "igbops"


def run_subprocess_command(command_list):
    try:
        result = subprocess.run(command_list, check=True, capture_output=True, text=True, errors='ignore')
        # print(f"CMD OK: {' '.join(command_list)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"CMD FAILED: {' '.join(e.cmd)}\nExit {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"CMD FAILED: Executable not found: {command_list[0]}")
        return False


def calculate_igb_difference(input_x_igb, input_y_igb, output_diff_igb, fibrotic_delay=0, healthy_delay=0):
    if not (os.path.exists(input_x_igb) and os.path.exists(input_y_igb)):
        return False
    ops = "X-Y"
    if fibrotic_delay != 0 or healthy_delay != 0:
        ops = f"X-Y-{fibrotic_delay}+{healthy_delay}"
    cmd = [IGBOPS_EXEC, "--expr", ops, input_x_igb, input_y_igb, "-O", output_diff_igb]
    return run_subprocess_command(cmd)


def _load_activation_data(igb_file_path):
    """Helper to load a single activation map from an IGB file."""
    try:
        _, raw_data, header = igb_reader(igb_file_path)
        if raw_data is not None and raw_data.shape[0] > 0 and raw_data.shape[1] > 0:  # Check if data exists
            # Assuming activation time is the first (or only) time slice
            return raw_data[0, :]
        print(
            f"Warning: No valid data found in {os.path.basename(igb_file_path)} (t_dim={header.get('t_dim', 'N/A')}, n_nodes={header.get('n_nodes', 'N/A')}).")
    except Exception as e:  # Catch errors from igb_reader (RuntimeError, FileNotFoundError for its tools, ValueError)
        print(f"Error reading IGB {os.path.basename(igb_file_path)}: {e}")
    return None


def evaluate_fibrosis_detection(
    model,
    healthy_parquet_sim_id,
    ecg_parquet_file_abs_path,
    block_pts_file_abs_path,
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
    metrics_direct_fib_pcc = []  # New metric list
    metrics_direct_fib_cosine = []  # New metric list

    healthy_parquet_sim_id = int(healthy_parquet_sim_id)
    print(f"--- Evaluation: {model_name_tag} | Healthy Parquet ID: {healthy_parquet_sim_id} ---")

    if not all(os.path.exists(p) for p in [ecg_parquet_file_abs_path, block_pts_file_abs_path, sim_raw_base_abs_path]):
        print("ERROR: Essential input file/directory not found. Aborting.")
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

    healthy_sim_raw_dir_id_str = str(healthy_parquet_sim_id // 10)
    healthy_sim_output_dir = os.path.join(sim_raw_base_abs_path, healthy_sim_raw_dir_id_str)
    os.makedirs(healthy_sim_output_dir, exist_ok=True)
    sim_healthy_act_path = os.path.join(healthy_sim_output_dir, "act.igb")
    if not os.path.exists(sim_healthy_act_path):
        print(f"ERROR: Ground truth healthy act.igb not found: {sim_healthy_act_path}. Aborting.")
        return None

    # --- Model Prediction for HEALTHY Heart ---
    model_pred_healthy_igb_path = os.path.join(
        healthy_sim_output_dir, f"model_pred_healthy_parquetID_{healthy_parquet_sim_id}_{model_name_tag}.igb")
    if force_regenerate_predictions or not os.path.exists(model_pred_healthy_igb_path):
        print(f"Predicting HEALTHY: Parquet ID {healthy_parquet_sim_id}")
        temp_dat_path = model_pred_healthy_igb_path.replace(".igb", ".dat")
        write_pinnex_predictions_streamed(
            model, block_pts_file_abs_path, ecg_parquet_file_abs_path, healthy_parquet_sim_id,
            temp_dat_path, prediction_batch_size, CV_predictions
        )
        # write_pinnex_predictions_streamed should handle .dat to .igb conversion.
        if not os.path.exists(model_pred_healthy_igb_path):  # Double check
            expected_igb_from_func = temp_dat_path.replace(".dat", ".igb")
            if os.path.exists(expected_igb_from_func) and expected_igb_from_func != model_pred_healthy_igb_path:
                shutil.move(expected_igb_from_func, model_pred_healthy_igb_path)
            else:
                print(f"ERROR: Failed to generate model healthy prediction: {model_pred_healthy_igb_path}. Aborting.")
                return None
    # ---

    fibrotic_parquet_sim_ids = [pid for pid in all_parquet_sim_ids if pid // 10 != healthy_parquet_sim_id // 10]
    # fibrotic_parquet_sim_ids = [pid for pid in fibrotic_parquet_sim_ids if pid // 10 < 20]  # Only even IDs

    if not fibrotic_parquet_sim_ids:
        print("No fibrotic simulations to process.")
        return {}

    cached_simulated_diff_paths = {}

    # --- Read Absolute Delay File ---
    if not os.path.exists(delay_file_abs_path):
        print(f"ERROR: Delay file not found: {delay_file_abs_path}. Aborting.")
        return None

    try:
        with open(delay_file_abs_path, 'r') as delay_file:
            delay_data = delay_file.readlines()
            print(f"Loaded delay file with {len(delay_data)} lines.")
    except Exception as e:
        print(f"ERROR reading delay file {delay_file_abs_path}: {e}. Aborting.")
        return None

    delay_map = {}
    for line in delay_data:
        parquet_id, delay = line.strip().split(',')
        delay_map[int(parquet_id)] = float(delay)

    for fib_parquet_sim_id in fibrotic_parquet_sim_ids:
        print(f"\nProcessing FIBROTIC: Parquet ID {fib_parquet_sim_id}")
        fib_sim_raw_dir_id_str = str(fib_parquet_sim_id // 10)
        current_sim_proc_dir = os.path.join(sim_raw_base_abs_path, fib_sim_raw_dir_id_str)
        os.makedirs(current_sim_proc_dir, exist_ok=True)

        sim_fibrotic_act_path = os.path.join(current_sim_proc_dir, "act.igb")
        if not os.path.exists(sim_fibrotic_act_path):
            print(f"  Skipping: Ground truth fibrotic act.igb not found: {sim_fibrotic_act_path}")
            continue

        # --- Simulated Difference (Ground Truth) ---
        sim_diff_cache_key = (fib_sim_raw_dir_id_str, healthy_sim_raw_dir_id_str)
        diff_simulated_igb_path = cached_simulated_diff_paths.get(sim_diff_cache_key)
        if not diff_simulated_igb_path or force_regenerate_diffs or not os.path.exists(diff_simulated_igb_path):
            diff_sim_filename = f"diff_simulated_base{fib_sim_raw_dir_id_str}_vs_base{healthy_sim_raw_dir_id_str}.igb"
            diff_simulated_igb_path = os.path.join(current_sim_proc_dir, diff_sim_filename)
            if force_regenerate_diffs or not os.path.exists(diff_simulated_igb_path):
                if not calculate_igb_difference(sim_fibrotic_act_path, sim_healthy_act_path, diff_simulated_igb_path):
                    print(f"  Skipping: Failed to create simulated difference for base {fib_sim_raw_dir_id_str}")
                    continue
            cached_simulated_diff_paths[sim_diff_cache_key] = diff_simulated_igb_path
        # ---

        # --- Model Prediction for FIBROTIC Heart ---
        model_pred_fibrotic_igb_path = os.path.join(
            current_sim_proc_dir, f"model_pred_fibrotic_parquetID_{fib_parquet_sim_id}_{model_name_tag}.igb")
        if force_regenerate_predictions or not os.path.exists(model_pred_fibrotic_igb_path):
            temp_dat_path = model_pred_fibrotic_igb_path.replace(".igb", ".dat")
            write_pinnex_predictions_streamed(
                model, block_pts_file_abs_path, ecg_parquet_file_abs_path, fib_parquet_sim_id,
                temp_dat_path, prediction_batch_size, CV_predictions
            )
            if not os.path.exists(model_pred_fibrotic_igb_path):  # Double check
                expected_igb_from_func = temp_dat_path.replace(".dat", ".igb")
                if os.path.exists(expected_igb_from_func) and expected_igb_from_func != model_pred_fibrotic_igb_path:
                    shutil.move(expected_igb_from_func, model_pred_fibrotic_igb_path)
                else:
                    print(
                        f"  Skipping: Failed to generate model fibrotic prediction for Parquet ID {fib_parquet_sim_id}")
                    continue
        # ---

        # --- Model-Predicted Difference ---
        diff_model_pred_igb_path = os.path.join(
            current_sim_proc_dir, f"diff_model_pred_parquetID_{fib_parquet_sim_id}_vs_healthy_{healthy_parquet_sim_id}_{model_name_tag}.igb")
        if force_regenerate_diffs or not os.path.exists(diff_model_pred_igb_path):
            fibrotic_delay = delay_map[fib_parquet_sim_id]
            healthy_delay = delay_map[healthy_parquet_sim_id]
            if not calculate_igb_difference(model_pred_fibrotic_igb_path, model_pred_healthy_igb_path, diff_model_pred_igb_path, fibrotic_delay=fibrotic_delay, healthy_delay=healthy_delay):
                print(f"  Skipping: Failed to create model-predicted difference for Parquet ID {fib_parquet_sim_id}")
                continue
        # ---

        # --- Load data for METRICS ---
        data_sim_diff = _load_activation_data(diff_simulated_igb_path)
        data_model_diff = _load_activation_data(diff_model_pred_igb_path)

        data_sim_diff_mean = np.mean(data_sim_diff)
        data_sim_indices = np.where(data_sim_diff > data_sim_diff_mean)[0]
        data_model_diff_mean = np.mean(data_model_diff)
        data_model_indices = np.where(data_model_diff > data_model_diff_mean)[0]

        data_sim_diff = data_sim_diff
        data_model_diff = data_model_diff

        union_indices = np.union1d(data_sim_indices, data_model_indices)

        data_sim_diff = data_sim_diff[union_indices]
        data_model_diff = data_model_diff[union_indices]

        data_sim_fibrotic_raw = _load_activation_data(sim_fibrotic_act_path)  # Ground truth fibrotic
        data_model_pred_fibrotic_raw = _load_activation_data(
            model_pred_fibrotic_igb_path)  # Model's fibrotic prediction

        if data_sim_diff is None or data_model_diff is None:
            print(
                f"  Skipping diff metrics for Parquet ID {fib_parquet_sim_id} due to loading error for difference files.")
        elif data_sim_diff.shape != data_model_diff.shape:
            print(
                f"  Skipping diff metrics for Parquet ID {fib_parquet_sim_id} due to shape mismatch in difference files.")
        else:
            pcc, _ = pearsonr(data_sim_diff, data_model_diff)
            cos_sim_diff = cosine_similarity_1d(data_sim_diff, data_model_diff)
            metrics_diff_pcc.append(pcc)
            metrics_diff_cosine.append(cos_sim_diff)
            print(f"  Diff Metrics - PCC: {pcc:.4f}, CosineSim(Diff): {cos_sim_diff:.4f}")

        if data_sim_fibrotic_raw is None or data_model_pred_fibrotic_raw is None:
            print(
                f"  Skipping direct metrics for Parquet ID {fib_parquet_sim_id} due to loading error for raw/predicted fibrotic files.")
        elif data_sim_fibrotic_raw.shape != data_model_pred_fibrotic_raw.shape:
            print(
                f"  Skipping direct metrics for Parquet ID {fib_parquet_sim_id} due to shape mismatch in raw/predicted fibrotic files.")
        else:
            pcc_fib, _ = pearsonr(data_sim_fibrotic_raw, data_model_pred_fibrotic_raw)
            metrics_direct_fib_pcc.append(pcc_fib)
            print(f"  Direct Fibrotic PCC: {pcc_fib:.4f}")
            cos_sim_direct_fib = cosine_similarity_1d(data_sim_fibrotic_raw, data_model_pred_fibrotic_raw)
            metrics_direct_fib_cosine.append(cos_sim_direct_fib)
            print(f"  Direct Fibrotic CosineSim: {cos_sim_direct_fib:.4f}")

        print(f"progress: {len(metrics_diff_pcc)}/{len(fibrotic_parquet_sim_ids)}")

    # --- Final Averaged Metrics ---
    results = {
        "fibrotic_parquet_sim_ids": fibrotic_parquet_sim_ids,
        "avg_diff_pcc": np.nanmean(metrics_diff_pcc) if metrics_diff_pcc else np.nan,
        "avg_diff_cosine": np.nanmean(metrics_diff_cosine) if metrics_diff_cosine else np.nan,
        "avg_direct_fib_pcc": np.nanmean(metrics_direct_fib_pcc) if metrics_direct_fib_pcc else np.nan,
        "avg_direct_fib_cosine": np.nanmean(metrics_direct_fib_cosine) if metrics_direct_fib_cosine else np.nan,
        "individual_diff_pcc": metrics_diff_pcc,
        "individual_diff_cosine": metrics_diff_cosine,
        "individual_direct_fib_pcc": metrics_direct_fib_pcc,
        "individual_direct_fib_cosine": metrics_direct_fib_cosine,
    }
    print("\n--- Averaged Results ---")
    for key, val in results.items():
        if "avg_" in key:
            print(f"{key}: {val:.4f}")
    print("--- Evaluation Complete ---")
    return results
