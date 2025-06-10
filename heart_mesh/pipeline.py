#!/usr/bin/env python3
import os, time
import vtk, subprocess, shutil            # same external deps

# all heavy helpers live here
from helper import (
    replace_class_label, run_cobiveco, run_fibers,
    prepare_openCarp, run_set_fibers
)

def main() -> None:
    folder              = "1600"
    cobiveco_folder     = "/Users/jesperarnwald/Documents/GitHub/Cobiveco/Heart"
    fiber_folder        = "/Users/jesperarnwald/Documents/GitHub/LDRB_Fibers/Heart"
    volume_file_name    = "heart_vol.vtu"
    surface_file_name   = "heart_sur.vtp"

    start_total = time.time()

    mesh_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), "Meshes", folder))
    surface_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "Meshes"))
    input_surface = os.path.join(surface_dir, surface_file_name)
    modified_surface = os.path.join(surface_dir, "heart_sur_4regions.vtp")
    volume_file     = os.path.join(mesh_dir, volume_file_name)

    # ------------ 1) Replace classes on the surface -----------------
    print("Step 1: Replacing class label …")
    replace_class_label(input_surface, modified_surface)    # un-comment when ready
    print(f"Done in {time.time()-start_total:.2f}s\n")

    # ------------ 2) Cobiveco coordinates ---------------------------
    print("Step 2: Running cobiveco …")
    run_cobiveco(volume_file, modified_surface, cobiveco_folder)
    print(f"Done in {time.time()-start_total:.2f}s\n")

    # ------------ 3) Fibre assignment -------------------------------
    print("Step 3: Running fibres …")
    run_fibers(volume_file, input_surface, fiber_folder)
    print(f"Done in {time.time()-start_total:.2f}s\n")

    # ------------ 4) openCARP conversion ----------------------------
    print("Step 4: prepare_openCarp …")
    prepare_openCarp(folder, os.path.join(mesh_dir, "cobiveco.vtu"))
    print(f"Done in {time.time()-start_total:.2f}s\n")

    # ------------ 5) Fibre export to .lon ---------------------------
    print("Step 5: set_fibers …")
    run_set_fibers(folder)
    print(f"Done in {time.time()-start_total:.2f}s\n")

    # ------------ 6) Electrodes -------------------------------------
    from put_electrodes import run_electrodes
    print("Step 6: put_electrodes …")
    run_electrodes(folder)
    print(f"Done in {time.time()-start_total:.2f}s\n")

    # ------------ 7) Region masking ---------------------------------
    from set_regions import run_set_regions
    print("Step 7: set_regions …")
    run_set_regions(folder)
    print(f"Done in {time.time()-start_total:.2f}s\n")

    print(f"Full pipeline finished in {time.time()-start_total:.2f}s")

if __name__ == "__main__":
    main()
