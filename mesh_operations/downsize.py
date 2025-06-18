#!/usr/bin/env python3
"""
downsize.py
===========

Down-samples a tetrahedral mesh using meshtool.

Steps
-----
1. Locate the original VTU surface file (heart_vol.vtu or instance_<folder>.vtu)
   in <project-root>/Meshes/<folder>.
2. Convert the VTU file to legacy VTK (ASCII) for meshtool.
3. Run meshtool “resample mesh” with a target average edge length.
4. Save the result in <project-root>/Meshes/<avrg> (e.g. 1200).
5. Convert the resampled VTK back to VTU.

Usage
-----
python downsize.py --folder 1600 --avrg 1200
python downsize.py --folder Mean --avrg 2000
"""

import argparse
import os
import subprocess
import sys
import vtk


# -----------------------------------------------------------------------------#
# Conversion helpers
# -----------------------------------------------------------------------------#
def convert_vtu_to_vtk(input_vtu: str, output_vtk: str) -> None:
    """Convert a VTU file to a legacy VTK file (ASCII)."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_vtu)
    reader.Update()

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(reader.GetOutput())
    writer.SetFileName(output_vtk)
    writer.SetFileTypeToASCII()
    writer.Write()

    print(f"[INFO] Converted '{input_vtu}' → '{output_vtk}'")


def convert_vtk_to_vtu(input_vtk: str, output_vtu: str) -> None:
    """Convert a legacy VTK file back to VTU."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_vtk)
    reader.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(reader.GetOutput())
    writer.SetFileName(output_vtu)
    writer.Write()

    print(f"[INFO] Converted '{input_vtk}' → '{output_vtu}'")


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Down-sample a tetrahedral mesh")
    p.add_argument(
        "--heart",
        required=True,
        help="mesh folder inside ‘Meshes’, e.g. Mean or 001",
    )
    p.add_argument(
        "--folder",
        required=True,
        help="mesh folder inside ‘Heart folder’, e.g. 1600 or Mean",
    )
    p.add_argument(
        "--avrg",
        type=int,
        required=True,
        help="target average edge length in µm, e.g. 1200",
    )
    return p


# -----------------------------------------------------------------------------#
# Main routine
# -----------------------------------------------------------------------------#
def main(folder: str, avrg_edge: int, heart: str) -> None:
    """Run the down-sampling pipeline."""
    # --------------------------------------------------------------------- #
    # Resolve paths
    # --------------------------------------------------------------------- #
    script_dir = os.path.dirname(__file__)
    heart_root = os.path.abspath(os.path.join(script_dir, "..", "Meshes", heart))

    base_dir = os.path.join(heart_root, folder)
    out_folder = os.path.join(heart_root, str(avrg_edge))
    os.makedirs(out_folder, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Locate input VTU
    # --------------------------------------------------------------------- #

    input_vtu = os.path.join(base_dir, "heart_vol.vtu")

    if not os.path.exists(input_vtu):
        sys.exit(f"[ERROR] Input file '{input_vtu}' not found.")

    # --------------------------------------------------------------------- #
    # Convert VTU → VTK
    # --------------------------------------------------------------------- #
    intermediate_vtk = os.path.join(out_folder, "heart_vol.vtk")
    convert_vtu_to_vtk(input_vtu, intermediate_vtk)

    input_basename = os.path.splitext(intermediate_vtk)[0]
    output_basename = os.path.join(out_folder, "heart_vol")

    # --------------------------------------------------------------------- #
    # meshtool resample
    # --------------------------------------------------------------------- #
    avrg_mm = avrg_edge / 1000.0  # convert µm → mm for meshtool

    meshtool_cmd = [
        "meshtool",
        "resample",
        "mesh",
        f"-msh={input_basename}",
        f"-avrg={avrg_mm}",
        f"-outmsh={output_basename}",
        "-ifmt=vtk",
        "-ofmt=vtk",
    ]

    print("[INFO] Running meshtool:")
    print("       " + " ".join(meshtool_cmd))
    try:
        subprocess.run(meshtool_cmd, check=True)
    except subprocess.CalledProcessError:
        sys.exit("[ERROR] meshtool resample failed.")

    output_vtk = output_basename + ".vtk"
    if not os.path.exists(output_vtk):
        sys.exit(f"[ERROR] meshtool output '{output_vtk}' not found.")

    # --------------------------------------------------------------------- #
    # Convert VTK → VTU
    # --------------------------------------------------------------------- #
    output_vtu = output_basename + ".vtu"
    convert_vtk_to_vtu(output_vtk, output_vtu)
    for fp in (
        intermediate_vtk,
        output_vtk,
        output_basename + ".fcon",  # meshtool continuity file
    ):
        if os.path.exists(fp):
            try:
                os.remove(fp)
                print(f"[CLEAN] Removed '{fp}'")
            except OSError as err:
                print(f"[WARN] Could not remove '{fp}': {err}")

    print("[DONE] Downsizing completed successfully.")


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.folder, args.avrg, args.heart)
