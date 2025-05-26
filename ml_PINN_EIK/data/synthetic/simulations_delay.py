import os
import subprocess
import random
import shutil
import json

#!/usr/bin/env python3


def generate_stimulation_params(n_simulations=2):
    """
    Generate a list of simulation parameter sets.
    Each parameter set is represented as a dictionary.

    The stimulation delay is randomly chosen between 0 and 30 ms.
    n_simulations: Total number of simulation configurations to generate.
    """
    param_list = []
    for i in range(n_simulations):
        delay = 0  # random.uniform(0, 0)

        if random.choice([True, False]):
            fibrosis = 0
        else:
            fibrosis = 0  # 0.2

        params = {
            "sim_id": str(i + 1),
            "stim_delay": delay,
            "fibrosis_vol_fraction": fibrosis,
            # You can add more parameters here if needed.
        }
        param_list.append(params)
    return param_list


def process_simulation_output(sim_id, proto_dir, target_dir, file_list):
    """
    Copy specified files from the simulation folder (named sim_id) in proto_dir 
    to target_dir, then remove the simulation folder.

    Parameters:
        sim_id (str): The simulation folder name.
        proto_dir (str): The directory containing simulation folders.
        target_dir (str): The directory to copy files to.
        file_list (list): List of file names to copy.
    """
    sim_folder = os.path.join(proto_dir, sim_id)
    mesh_folder = os.path.join(proto_dir, "mesh")

    if not os.path.exists(sim_folder):
        print(f"Simulation folder {sim_folder} does not exist!")
        return

    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Copy each file in file_list from sim_folder to target_dir
    for file_name in file_list:
        src = os.path.join(sim_folder, file_name)
        dst = os.path.join(target_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File {src} does not exist, skipping.")

    # Remove the simulation folder after copying the files
    shutil.rmtree(sim_folder)
    print(f"Removed simulation folder {sim_folder}")

    if os.path.exists(mesh_folder):
        shutil.rmtree(mesh_folder)
        print(f"Removed mesh folder {mesh_folder}")
    else:
        print(f"Mesh folder {mesh_folder} not found, skipping.")


def run_simulations():
    """
    Run the ml_fibrosis.py simulation script for each generated parameter set.
    The simulation ID will be the set number (starting at 1).
    After each simulation, the helper function copies the required files into
    current_dir/sim and then removes the simulation folder.
    """
    # Generate the parameter list
    param_list = generate_stimulation_params(n_simulations=1)

    # Print the parameter list for inspection
    print("Generated Stimulation Parameters:")
    for idx, params in enumerate(param_list, start=1):
        print(f"Set {idx}: {params}")

    # Current directory (where the simulations.py is run)
    current_dir = os.getcwd()

    # Create the main output directory for simulations
    sim_raw_dir = os.path.join(current_dir, "sim_raw")
    os.makedirs(sim_raw_dir, exist_ok=True)

    # Define the settings file path
    settings_file = os.path.join(sim_raw_dir, "simulation_settings.json")

    # Move three folders up and then into the prototype folder
    proto_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "prototype/ml_EIK"))
    simulation_script = os.path.join(proto_dir, "ml_eik_fibrosis.py")

    # List of file names (these names should be the same for all simulations)
    file_list = ["act.igb", "healthy_ECG.csv"]

    # Run one simulation per parameter set
    for params in param_list:
        sim_id = params["sim_id"]
        target_dir = os.path.join(sim_raw_dir, sim_id)

        # Build the command
        cmd = [
            "python3.10", simulation_script,
            "--stim_delay", str(params["stim_delay"]),
            "--simID", sim_id,
            "--ECG", sim_id,  # Adjust if needed; here we pass sim_id as ECG identifier.
            "--fibrosis_vol_fraction", str(params["fibrosis_vol_fraction"]),
        ]
        print(f"Running simulation: {sim_id} with stimulation delay {params['stim_delay']} ms")
        subprocess.run(cmd, check=True, cwd=proto_dir)
        cmd2 = [
            "python", simulation_script,
            "--visualize"
        ]
        # subprocess.run(cmd2, check=True, cwd=proto_dir)

        # After the simulation completes, process its output:
        process_simulation_output(sim_id, proto_dir, target_dir, file_list)

    # Save simulation settings to JSON file
    with open(settings_file, "w") as f:
        json.dump(param_list, f, indent=4)

    print(f"Saved simulation settings to {settings_file}")


if __name__ == "__main__":
    run_simulations()
