#!/usr/bin/env python3
import os, time
from mesh_io         import replace_class_label
from step_cobiveco   import run_cobiveco
from step_fibers     import run_fibers
from step_opencarp   import prepare_openCarp, run_set_fibers
from step_electrodes import run_electrodes
from step_regions    import run_set_regions


import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="heart-mesh pipeline")
    p.add_argument("folder", help="mesh folder name, e.g. 1600")

    # run-mode switches
    p.add_argument("--electrodes", action="store_true",
                   help="run ONLY the electrode-placement step")
    p.add_argument("--regions", action="store_true",
                   help="run ONLY the region-masking step "
                        "(add --electrodes to run both)")

    return p



def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    folder = args.folder

    # ----------------------------------------------------------------
    # paths (unchanged lines below; kept here for context)
    # ----------------------------------------------------------------
    cobiveco_folder  = "/Users/jesperarnwald/Documents/GitHub/Cobiveco/Heart"
    fiber_folder     = "/Users/jesperarnwald/Documents/GitHub/LDRB_Fibers/Heart"
    volume_file_name = "heart_vol.vtu"
    surface_file_name = "heart_sur.vtp"

    start_total = time.time()

    mesh_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   "Meshes", folder))
    surface_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   "Meshes"))
    input_surface   = os.path.join(surface_dir, surface_file_name)
    modified_surface = os.path.join(surface_dir, "heart_sur_4regions.vtp")
    volume_file     = os.path.join(mesh_dir, volume_file_name)

    # ----------------------------------------------------------------
    # 1 – 5  run **only** when no step-specific flag is given
    # ----------------------------------------------------------------
    if not (args.electrodes or args.regions):
        print("Step 1: Replacing class label …")
        replace_class_label(input_surface, modified_surface)
        print(f"Done in {time.time()-start_total:.2f}s\n")

        print("Step 2: Running cobiveco …")
        run_cobiveco(volume_file, modified_surface, cobiveco_folder)
        print(f"Done in {time.time()-start_total:.2f}s\n")

        print("Step 3: Running fibres …")
        run_fibers(volume_file, input_surface, fiber_folder)
        print(f"Done in {time.time()-start_total:.2f}s\n")

        print("Step 4: prepare_openCarp …")
        prepare_openCarp(folder, os.path.join(mesh_dir, "cobiveco.vtu"))
        print(f"Done in {time.time()-start_total:.2f}s\n")

        print("Step 5: set_fibers …")
        run_set_fibers(folder)
        print(f"Done in {time.time()-start_total:.2f}s\n")

    # ----------------------------------------------------------------
    # 6 – 7  → controlled by CLI flags
    # ----------------------------------------------------------------
    if args.electrodes:
        print("Step 6: put_electrodes …")
        run_electrodes(folder)
        print(f"Done in {time.time()-start_total:.2f}s\n")

    if args.regions:
        print("Step 7: set_regions …")
        run_set_regions(folder)
        print(f"Done in {time.time()-start_total:.2f}s\n")

    print(f"Full pipeline finished in {time.time()-start_total:.2f}s")



if __name__ == "__main__":
    main()
