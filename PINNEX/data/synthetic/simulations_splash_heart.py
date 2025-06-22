#!/usr/bin/env python3
import os
import subprocess
import random
import shutil
import json
import math


def generate_stimulation_params(n_simulations=2):
    """
    Generate a list of simulation parameter sets, each containing:
      - A random center for the fibrotic region (fibrosis_center_x/y/z).
      - A random radius/size (fibrosis_size).
      - Optionally, other parameters like stim_delay, fibrosis_vol_fraction.
    """
    param_list = []
    for i in range(n_simulations):

        params = {
            "sim_id": str(i + 1),
            "n_patches": 1,
            "random_seed": str(i + 1),
            "mesh": str(1600),
            "heart": "Mean",
        }
        param_list.append(params)
    return param_list


def process_simulation_output(sim_id, proto_dir, target_dir, file_list):
    """
    Copy specified files from the simulation folder (named sim_id) in proto_dir
    to target_dir, then remove the simulation folder and the 'mesh' folder.
    """
    sim_folder = os.path.join(proto_dir, sim_id)
    mesh_folder = os.path.join(proto_dir, "mesh")

    if not os.path.exists(sim_folder):
        print(f"Simulation folder '{sim_folder}' does not exist!")
        return

    os.makedirs(target_dir, exist_ok=True)
    for file_name in file_list:
        src = os.path.join(sim_folder, file_name)
        dst = os.path.join(target_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied '{src}' to '{dst}'.")
        else:
            print(f"File '{src}' not found; skipping.")

    shutil.rmtree(sim_folder)
    print(f"Removed simulation folder '{sim_folder}'.")

    if os.path.exists(mesh_folder):
        shutil.rmtree(mesh_folder)
        print(f"Removed mesh folder '{mesh_folder}'.")
    else:
        print(f"Mesh folder '{mesh_folder}' not found; skipping.")


def run_simulations(n_simulations=2):
    """
    For each parameter set:
      - Run the simulation script with extra arguments for fibrotic center and size.
      - Process and store the simulation outputs.
      - Save the simulation settings as JSON.
    """
    param_list = generate_stimulation_params(n_simulations)
    print("Generated Stimulation Parameters:")
    for idx, params in enumerate(param_list, start=1):
        print(f"Set {idx}: {params}")

    current_dir = os.getcwd()
    sim_raw_dir = os.path.join(current_dir, "sim_raw")
    os.makedirs(sim_raw_dir, exist_ok=True)
    settings_file = os.path.join(sim_raw_dir, "simulation_settings.json")

    # Point to your actual script location
    proto_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "ml_Heart"))
    simulation_script = os.path.join(proto_dir, "splash_fibrosis_heart_realistic.py")

    # List of file names to be collected
    file_list = ["act.igb", "ecg.csv", "point_cv_stats.csv"]

    for params in param_list:
        sim_id = params["sim_id"]
        target_dir = os.path.join(sim_raw_dir, sim_id)

        cmd = [
            "python3",
            simulation_script,
            "--duration",
            str(200),
            "--simID",
            sim_id,
            "--random_seed",
            str(params["random_seed"]),
            "--n_patches",
            str(params["n_patches"]),
            "--mesh",
            str(params["mesh"]),
            "--heart",
            params["heart"],
        ]
        print(" ".join(cmd))
        print(f"Running simulation '{sim_id}' ...")
        subprocess.run(cmd, check=True, cwd=proto_dir)

        # Uncomment if you want to do additional steps or visualization
        # cmd_visualize = ["python", simulation_script, "--visualize"]
        # subprocess.run(cmd_visualize, check=True, cwd=proto_dir)

        # Gather results and clean up
        process_simulation_output(sim_id, proto_dir, target_dir, file_list)

        params_file = os.path.join(target_dir, "params.txt")
        with open(params_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

    # Save the final parameter list
    with open(settings_file, "w") as f:
        json.dump(param_list, f, indent=4)
    print(f"Saved simulation settings to '{settings_file}'.")


def main():
    run_simulations(n_simulations=1)  # or however many you'd like to run


if __name__ == "__main__":
    main()
