import torch


def eikonal_residual(
    T_pred,         # (N, 1) Model's prediction for activation time
    c_pred,         # (N, 1) Model's prediction for conduction velocity
    dT_dx,          # (N, 1) Pre-computed gradient of T_pred w.r.t. x
    dT_dy,          # (N, 1) Pre-computed gradient of T_pred w.r.t. y
    dT_dz,          # (N, 1) Pre-computed gradient of T_pred w.r.t. z
    epsilon=1e-8    # Small value for numerical stability
):
    """
    Computes the residual of the Eikonal equation: |∇T| - 1/c.
    This version assumes T_pred, c_pred, and spatial gradients of T_pred (dT_dx, dT_dy, dT_dz)
    are provided as inputs.

    Args:
      T_pred (torch.Tensor): Predicted activation time, shape (N, 1).
      c_pred (torch.Tensor): Predicted conduction velocity, shape (N, 1).
      dT_dx (torch.Tensor): Gradient of T_pred w.r.t. x, shape (N, 1).
      dT_dy (torch.Tensor): Gradient of T_pred w.r.t. y, shape (N, 1).
      dT_dz (torch.Tensor): Gradient of T_pred w.r.t. z, shape (N, 1).
      epsilon (float): Small constant for numerical stability (e.g., in sqrt and division).

    Returns:
      torch.Tensor: The residual of the Eikonal equation, shape (N, 1).
    """

    # 1) Ensure all necessary inputs are provided
    if T_pred is None:
        raise ValueError("T_pred must be provided.")
    if c_pred is None:
        # If c_pred is essential for the residual and not provided, this is an issue.
        # Depending on the Eikonal formulation, 1/c might be replaced or c handled differently.
        # For this standard form, c_pred is required.
        raise ValueError("c_pred must be provided for the Eikonal residual calculation.")
    if any(g is None for g in [dT_dx, dT_dy, dT_dz]):
        raise ValueError("Spatial gradients (dT_dx, dT_dy, dT_dz) must be provided.")

    # 2) Compute the gradient magnitude |∇T| from pre-computed gradients
    grad_T_norm_sq = dT_dx**2 + dT_dy**2 + dT_dz**2
    grad_mag_T = torch.sqrt(grad_T_norm_sq + epsilon)

    # 3) Build the Eikonal PDE residual: |∇T| - 1/c
    # Add epsilon to c_pred in the denominator for numerical stability.
    # c_pred is expected to be positive.
    residual = grad_mag_T - 1.0 / (c_pred + epsilon)

    return residual
