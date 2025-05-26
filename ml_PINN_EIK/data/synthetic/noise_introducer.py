#!/usr/bin/env python3
"""
Script to add Gaussian noise to activation times in a parquet file.

Requirements:
- pandas
- numpy
- pyarrow (for parquet support)

Install with:
pip install pandas numpy pyarrow
"""

import pandas as pd
import numpy as np
import os
import argparse


def add_noise_to_activation_times(file_path, noise_percentage):
    """
    Reads a parquet file with activation times, adds Gaussian noise,
    and saves to a new file with the noise level indicated in the filename.

    Args:
        file_path (str): Path to the original parquet file
        noise_percentage (float): Percentage of max activation time to use as std dev
                                 (e.g., 0.1 means 10% of max activation time)
    """
    # Read the parquet file
    print(f"Reading file: {file_path}")
    df = pd.read_parquet(file_path)

    # Print the header of the DataFrame
    print("\nDataFrame Header:")
    print(df.head())
    print("\n")

    # Display info about the dataframe
    print(f"Original dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Find the activation time column (assuming it's named 'T')
    # If it has a different name, adjust accordingly
    if "T" in df.columns:
        activation_col = "T"
    else:
        # Try to find a column that might contain activation times
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Numeric columns: {numeric_cols.tolist()}")
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the dataframe")
        # Use the last numeric column as a guess (adjust if necessary)
        activation_col = numeric_cols[-1]
        print(f"Using {activation_col} as the activation time column")

    # Get the maximum activation time
    max_activation = df[activation_col].max()
    print(f"Maximum activation time: {max_activation}")

    # Calculate the standard deviation as a percentage of the maximum activation time
    std_dev = max_activation * noise_percentage
    print(
        f"Using std dev of {std_dev} ({noise_percentage*100:.1f}% of max activation time)"
    )

    # Create a copy of the dataframe
    df_noisy = df.copy()

    # Add Gaussian noise to the activation times
    noise = np.random.normal(0, std_dev, size=df.shape[0])
    df_noisy[activation_col] = df_noisy[activation_col] + noise
    max_activation_noisy = df_noisy[activation_col].max()
    print(f"Maximum noisy activation time: {max_activation_noisy}")

    # Make sure activation times remain non-negative
    if df_noisy[activation_col].min() < 0:
        print("Warning: Some noisy activation times became negative. Clipping to 0.")
        df_noisy[activation_col] = df_noisy[activation_col].clip(lower=0)

    # Create new filename with the noise level indicated
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_parts = os.path.splitext(base_name)
    new_filename = f"{name_parts[0]}_{noise_percentage:.2f}_noise{name_parts[1]}"
    output_path = os.path.join(dir_name, new_filename)

    # Save the noisy dataframe
    print(f"Saving noisy activation times to: {output_path}")
    df_noisy.to_parquet(output_path)

    return output_path, max_activation, std_dev


if __name__ == "__main__":
    # Set default path to the spatio_EIK_vol0 parquet file
    default_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "synthetic",
        "parsed",
        "spatio_EIK_vol0.parquet",
    )

    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to activation times in a parquet file"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=default_file_path,
        help=f"Path to the parquet file (default: {default_file_path})",
    )
    parser.add_argument(
        "--noise_percentage",
        type=float,
        default=0.01,
        help="Noise level as a percentage of max activation time (default: 0.01 = 1%)",
    )

    args = parser.parse_args()

    output_path, max_activation, std_dev = add_noise_to_activation_times(
        args.file_path, args.noise_percentage
    )

    print(f"\nSummary:")
    print(f"Original file: {args.file_path}")
    print(f"Max activation time: {max_activation}")
    print(
        f"Noise percentage: {args.noise_percentage:.2f} ({args.noise_percentage*100:.1f}%)"
    )
    print(f"Actual std dev used: {std_dev}")
    print(f"Output file: {output_path}")
