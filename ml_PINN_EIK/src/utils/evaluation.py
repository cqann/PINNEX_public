import numpy as np
import torch
import time


def evaluate_pinnex(wrapper, val_loader, device):
    """
    Evaluates the trained PINNWithECG model (for the eikonal equation) on the validation set.

    Args:
        model: The trained PINNWithECG model that predicts activation time T and conduction velocity c.
        val_loader: DataLoader containing validation data yielding:
                    (x, y, z, ecg, T, sim_ids)
        device: 'cuda' or 'cpu'

    Returns:
        true_T, pred_T: Ground truth vs. predicted activation time values.
        pred_c: Predicted conduction velocity values.
    """
    model = wrapper.model
    wrapper.start_new_epoch()  # Clear the ECG cache at the start of evaluation
    model.eval()  # Set model to evaluation mode
    true_T, pred_T, pred_c = [], [], []
    batch_n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_n += 1
            # Extract batch components
            x_val, y_val, z_val, ecg_val, T_val, C_val, sim_ids = [b.to(device) for b in batch]

            T_pred, c_pred = wrapper(x_val, y_val, z_val, ecg_val, sim_ids=sim_ids)
            # Append results (move tensors to CPU for processing)
            true_T.append(T_val.cpu().numpy())
            pred_T.append(T_pred.cpu().numpy())
            pred_c.append(c_pred.cpu().numpy())

            if batch_n * batch[0].shape[0] > 100000:
                break

    # Convert lists to numpy arrays
    true_T = np.concatenate(true_T, axis=0)
    pred_T = np.concatenate(pred_T, axis=0)
    pred_c = np.concatenate(pred_c, axis=0)

    return true_T, pred_T, pred_c


def compute_metrics(true_vals, pred_vals):
    """
    Computes Relative Error (RE), Mean Squared Error (MSE), and 
    Correlation Coefficient (CC) between predicted and true values.

    Args:
        true_vals: Ground truth values (numpy array)
        pred_vals: Model predictions (numpy array)

    Returns:
        re: Relative Error
        mse: Mean Squared Error
        cc: Correlation Coefficient
    """
    # Compute Relative Error (RE)
    re = np.sqrt(np.sum((pred_vals - true_vals) ** 2)) / np.sqrt(np.sum(true_vals ** 2))

    # Compute Mean Squared Error (MSE)
    mse = np.mean((pred_vals - true_vals) ** 2)

    # Compute Correlation Coefficient (CC)
    true_mean = np.mean(true_vals, axis=0, keepdims=True)
    pred_mean = np.mean(pred_vals, axis=0, keepdims=True)
    cc_numerator = np.sum((pred_vals - pred_mean) * (true_vals - true_mean))
    cc_denominator = np.sqrt(np.sum((pred_vals - pred_mean) ** 2)) * np.sqrt(np.sum((true_vals - true_mean) ** 2))
    cc = cc_numerator / cc_denominator

    return re, mse, cc
