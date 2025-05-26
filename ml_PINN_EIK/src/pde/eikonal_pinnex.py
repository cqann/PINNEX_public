import torch


def eikonal_residual(x, y, z, ecg, wrapper, params, sim_ids=None):
    """
    Compute the residual of the eikonal equation for electrocardiography of cardiomyocytes:
      |∇T| = 1/c,
    where T is the activation time predicted by the model and c is the conduction velocity.

    Inputs:
      x, y, z : torch.Tensors of shape (N,1) representing spatial coordinates.
      ecg     : torch.Tensor of shape (N, n_leads, seq_len) containing ECG context.
      wrapper : a PINN model whose forward pass predicts the activation time T; its forward expects (x, y, z, ecg)
      params  : dictionary containing PDE constants, e.g.:
                {
                  'c' : 0.6   # conduction velocity (in appropriate units)
                }
      sim_ids : optional simulation identifiers.

    Returns:
      residual : torch.Tensor of shape (N,1), the residual of the eikonal equation.
                 A perfect solution would yield residual = 0.
    """

    # 1) Forward pass: predict activation time T_pred (shape: (N,1))
    T_pred, c_pred = wrapper(x, y, z, ecg, sim_ids=sim_ids)  # shape: (N,1)

    # 2) PDE parameter: conduction velocity c and its inverse

    # 3) Compute spatial derivatives via autograd
    dT_dx = torch.autograd.grad(
        T_pred, x,
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True
    )[0]
    dT_dy = torch.autograd.grad(
        T_pred, y,
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True
    )[0]
    dT_dz = torch.autograd.grad(
        T_pred, z,
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True
    )[0]

    # 4) Compute the gradient magnitude |∇T|
    grad_mag = torch.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)

    # 5) Build the eikonal PDE residual: |∇T| * c - 1
    residual = grad_mag * c_pred - 1.0

    return residual


def check_eikonal_residual_with_trivial_solution(model_class, model_wrapper, params, device='cpu'):
    """
    Quick check to see if the eikonal PDE residual is near zero for a trivial solution.

    The chosen trivial solution is:
         T(x,y,z) = sqrt(x^2 + y^2 + z^2) / c,
    which analytically satisfies |∇T| = 1/c.
    """

    # Create a trivial model that implements the analytical solution.
    class TrivialEikonalModel(model_class):
        def forward(self, x, y, z, ecg):
            c = params['c']
            # Add a small epsilon for numerical stability at the origin.
            r = torch.sqrt(x**2 + y**2 + z**2 + 1e-6)
            return r / c

    trivial_model = model_wrapper(TrivialEikonalModel()).to(device)

    N = 10
    x = torch.rand(N, 1, device=device)
    y = torch.rand(N, 1, device=device)
    z = torch.rand(N, 1, device=device)

    # Ensure gradients are tracked for spatial variables.
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)

    # Create a dummy ECG tensor of shape (N, n_leads, seq_len)
    n_leads = 12
    seq_len = 1000
    ecg_dummy = torch.zeros(N, n_leads, seq_len, device=device)

    residual = eikonal_residual(x, y, z, ecg_dummy, trivial_model, params)

    print(f"Mean(|residual|): {residual.abs().mean().item():.6e}")
    return residual
