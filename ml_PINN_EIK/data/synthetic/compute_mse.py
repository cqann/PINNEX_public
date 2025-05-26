import numpy as np
import sys
import os
from igb_communication import igb_reader


def compute_metrics(true_file, pred_file):
    # Read the true values from act_cube.igb
    time_array_true, true_data, _ = igb_reader(true_file)
    # Take only one instance since they're all the same
    true_vals = true_data[0, :]  # Get activation times from first time step

    # Read the predicted values
    time_array_pred, pred_data, _ = igb_reader(pred_file)

    # Print shapes for debugging
    print("\nData shapes:")
    print(f"True data shape: {true_data.shape}")
    print(f"Prediction data shape: {pred_data.shape}")

    # If pred_data is 2D, take the first row
    pred_vals = pred_data[0, :] if len(pred_data.shape) > 1 else pred_data

    # Print value ranges for debugging
    print("\nValue ranges:")
    print(f"True values range: [{np.min(true_vals):.2f}, {np.max(true_vals):.2f}]")
    print(f"Pred values range: [{np.min(pred_vals):.2f}, {np.max(pred_vals):.2f}]")

    # Print first 10 values
    print("\nFirst 10 values comparison:")
    print("True values:", true_vals[:10])
    print("Pred values:", pred_vals[:10])
    print()

    # Compute MSE
    mse = np.mean((pred_vals - true_vals) ** 2)

    # Compute relative error
    re = np.sqrt(np.sum((pred_vals - true_vals) ** 2)) / np.sqrt(np.sum(true_vals**2))

    return mse, re


def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_mse.py <prediction_filename>")
        print("Note: The prediction file should be in the notebooks folder")
        sys.exit(1)

    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths
    true_file = os.path.join(script_dir, "raw/act_cube.igb")
    pred_file = os.path.join(script_dir, "../../notebooks", sys.argv[1])

    try:
        mse, re = compute_metrics(true_file, pred_file)
        print(f"\nComparing against file: {sys.argv[1]}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Relative Error: {re:.6f}")
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
