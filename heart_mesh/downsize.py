#!/usr/bin/env python3
"""
downsize.py
===========

This script downsamples a tetrahedral mesh using meshtool. It performs the following steps:
  1. Locate the original VTU surface file (heart_vol.vtu) in the "Mean" folder.
  2. Convert heart_vol.vtu to a legacy VTK file.
  3. Run meshtool's "resample mesh" command to resample the mesh based on an average edge length.
  4. Save the output in a folder named after the average edge length (e.g. "2000").
  5. Convert the resulting VTK file back to VTU format.

Ensure that:
  - meshtool is installed and available in your system PATH.
  - VTK Python modules are installed.
"""

import os
import subprocess
import sys
import vtk


def convert_vtu_to_vtk(input_vtu: str, output_vtk: str) -> None:
    """
    Converts a VTU file to a legacy VTK file in ASCII format.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_vtu)
    reader.Update()
    data = reader.GetOutput()

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_vtk)
    writer.SetInputData(data)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(f"Converted '{input_vtu}' to legacy VTK file '{output_vtk}'.")


def convert_vtk_to_vtu(input_vtk: str, output_vtu: str) -> None:
    """
    Converts a legacy VTK file to a VTU file.
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_vtk)
    reader.Update()
    data = reader.GetOutput()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_vtu)
    writer.SetInputData(data)
    writer.Write()
    print(f"Converted '{input_vtk}' to VTU file '{output_vtu}'.")


def main():
    # Define the base folder for the "Mean" mesh.
    # This assumes the folder structure: <project_root>/Meshes/Mean
    folder = "Mean"
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "Meshes", folder, "original"
        )
    )

    # Set the desired average edge length (for example, 2000)
    avrg = 1.2

    # Create an output folder named after the average edge length (e.g. "2000")
    out_folder = os.path.join(base_dir, "..", "1200")
    os.makedirs(out_folder, exist_ok=True)
    # Input VTU file: heart_vol.vtu should exist in the base_dir.
    if folder == "Mean":
        input_vtu = os.path.join(base_dir, f"heart_vol.vtu")
    else:
        input_vtu = os.path.join(base_dir, f"instance_{folder}.vtu")

    if not os.path.exists(input_vtu):
        print(f"Error: Input file '{input_vtu}' does not exist!")
        sys.exit(1)

    # Convert heart_vol.vtu to a legacy VTK file (needed for meshtool).
    intermediate_vtk = os.path.join(out_folder, f"instance_{folder}.vtk")
    convert_vtu_to_vtk(input_vtu, intermediate_vtk)

    # Prepare basenames for meshtool: remove the extension from the intermediate file.
    input_basename = os.path.splitext(intermediate_vtk)[0]
    # Define output basename: files will be created in out_folder.
    output_basename = os.path.join(out_folder, f"instance_{folder}")

    # Build the meshtool command.
    # This command passes the input mesh, average edge length, min/max edge sizes, and output basename.
    # Build the meshtool command with additional format options.
    meshtool_cmd = [
        "meshtool",
        "resample",
        "mesh",
        f"-msh={input_basename}",
        f"-avrg={avrg}",
        f"-outmsh={output_basename}",
        f"-ifmt=vtk",  # Set the input mesh format to vtk
        f"-ofmt=vtk",  # Set the output mesh format to vtk
        "-angl=30.0",
        "-surf_corr=0.95",
        "-postsmth=1",
        "-uniform=0",
        "-fix_bnd=0",
        "-conv=0",
    ]

    print("Running meshtool command:")
    print(" ".join(meshtool_cmd))
    try:
        subprocess.run(meshtool_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error: Meshtool resample mesh command failed.")
        sys.exit(1)

    # Meshtool produces an output VTK file with the basename and a .vtk extension.
    output_vtk = output_basename + ".vtk"
    if not os.path.exists(output_vtk):
        print(f"Error: Meshtool output file '{output_vtk}' not found.")
        sys.exit(1)

    # Convert the output VTK file back to a VTU file.
    output_vtu = output_basename + ".vtu"
    convert_vtk_to_vtu(output_vtk, output_vtu)

    print("Downsizing completed successfully.")


if __name__ == "__main__":
    main()
