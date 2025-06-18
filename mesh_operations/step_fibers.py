import os, shutil, subprocess

def run_fibers(volume_file, surface_file, matlab_folder):
    """
    Unchanged code from helper.run_fibers.
    """
    if not os.path.isdir(matlab_folder):
        raise ValueError(f"MATLAB folder '{matlab_folder}' does not exist.")

    input_folder  = os.path.join(matlab_folder, "Mesh")
    os.makedirs(input_folder, exist_ok=True)

    volume_dest  = os.path.join(input_folder, "heart.vtu")
    surface_dest = os.path.join(input_folder, "heart.vtp")

    shutil.copy(volume_file,  volume_dest)
    shutil.copy(surface_file, surface_dest)

    print(f"Copied volume file from '{volume_file}' to '{volume_dest}'")
    print(f"Copied surface file from '{surface_file}' to '{surface_dest}'")

    subprocess.run(["matlab", "-batch", "assign_fiber"], cwd=matlab_folder, check=True)

    result_file      = os.path.join(matlab_folder, "Result", "heart.vtu")
    dest_result_file = os.path.join(os.path.dirname(volume_file), "fiber.vtu")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file '{result_file}' not found.")
    shutil.copy(result_file, dest_result_file)
    print(f"Copied result file from '{result_file}' to '{dest_result_file}'")
