import os
from datetime import date

from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import testing
import numpy as np
from numpy import array as nplist
import matplotlib.pyplot as plt
from carputils import ep
from carputils.carpio import txt
import math
from scipy.signal import iirfilter, filtfilt
import pandas as pd  # <-- Import pandas
import random
import subprocess


def parser():
    parser = tools.standard_parser()
    group = parser.add_argument_group("experiment specific options")

    group.add_argument(
        "--duration",
        type=float,
        default=50.0,
        help="Duration of simulation (ms) (default is 50.)",
    )
    group.add_argument(
        "--simID",
        type=str,
        default=1.0,
        help="Duration of simulation (ms) (default is 50.)",
    )
    group.add_argument(
        "--mesh",
        type=str,
        default=1200,
        help="Duration of simulation (ms) (default is 50.)",
    )
    return parser


def jobID(args):
    return "CV_diff"


def run_command(command_list):
    """Helper function to run a command and print its details."""
    print(f"Executing: {' '.join(command_list)}")
    try:
        result = subprocess.run(
            command_list, check=True, capture_output=True, text=True
        )
        print("Command successful.")
        if result.stdout:
            print("Stdout:\n", result.stdout)
        if result.stderr:
            print("Stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Stdout:\n", e.stdout)
        if e.stderr:
            print("Stderr:\n", e.stderr)
        raise  # Re-raise the exception to stop the script if a command fails


@tools.carpexample(parser, jobID)
def run(args, job):
    CALLER_DIR = os.getcwd()
    print(CALLER_DIR)

    healthy_T = os.path.join(
        os.path.dirname(CALLER_DIR),
        "ml_Heart",
        args.mesh + "_heart",
        "act_healthy",
    )

    simID = args.simID
    healthy_igb_actual_path = healthy_T + ".igb"
    simulated_igb_path = os.path.join(simID, "act.igb")
    estimated_igb_path = os.path.join("..", "..", "pred_T_test.igb")

    difference_output_igb_estimated = os.path.join(
        simID, "difference_healthy_vs_estimated.igb"
    )
    difference_output_igb_simulated = os.path.join(
        simID, "difference_healthy_vs_simulated.igb"
    )

    igbops_diff_cmd = [
        settings.execs.igbops,
        "--expr",
        "X-Y",
        simulated_igb_path,  # "X"
        healthy_igb_actual_path,  # File 'Y'
        "-O",
        difference_output_igb_simulated,
    ]
    job.bash(igbops_diff_cmd)

    igbops_diff_cmd = [
        settings.execs.igbops,
        "--expr",
        "X-Y",
        estimated_igb_path,  # "X"
        healthy_igb_actual_path,  # File 'Y'
        "-O",
        difference_output_igb_estimated,
    ]
    job.bash(igbops_diff_cmd)

    estimated_CV = os.path.join("..", "..", "pred_T_test_physCV.igb")
    estimated_CV_healthy = os.path.join("..", "..", "pred_T_HealthyCV.igb")
    estimated_healthy = os.path.join("..", "..", "pred_T_Healthy.igb")
    diff_estimations = os.path.join(simID, "diff_estimated_healthy_vs_fib.igb")
    diff_estimations_CV = os.path.join(simID, "diff_estimated_healthy_vs_fib_CV.igb")

    igbops_diff_cmd = [
        settings.execs.igbops,
        "--expr",
        "X-Y",
        estimated_CV,  # "X"
        estimated_CV_healthy,  # File 'Y'
        "-O",
        diff_estimations_CV,
    ]
    job.bash(igbops_diff_cmd)

    igbops_diff_cmd = [
        settings.execs.igbops,
        "--expr",
        "X-Y",
        estimated_igb_path,  # "X"
        estimated_healthy,  # File 'Y'
        "-O",
        diff_estimations,
    ]
    job.bash(igbops_diff_cmd)


if __name__ == "__main__":
    run()
