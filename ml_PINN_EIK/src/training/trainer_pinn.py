# src/training/trainer_pinn.py

import torch
from torch import nn
import torch.optim as optim

from src.pde.aliev_panfilov import aliev_panfilov_residual


def train_pinn_minibatch(model,
                         train_loader,
                         val_loader,
                         params,
                         n_epochs=10000,
                         lr=1e-4,
                         physics_weight=1e-4,
                         batches_per_epoch=1000,
                         phys_batch_size=1000,
                         val_every_n_epochs=5,
                         device='cpu'):
    """
    Trains a PINN using mini-batches for data points and optional PDE collocation points.

    train_loader : DataLoader over (x,y,z,t,U,V) for training
    val_loader   : DataLoader over (x,y,z,t,U,V) for validation
    coords_phys  : PDE collocation points for the entire domain (or we sample in each epoch)
    params       : PDE constants dictionary
    """

    model = model.to(device)
    print(f"Training on {device}")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For logging
    train_loss_hist = []
    val_loss_hist = []
    data_loss_hist = []
    pde_loss_hist = []

    # We'll either:
    #   - keep coords_phys fixed,
    #   - or re-sample them each epoch. Here's a simple approach that re-samples if coords_phys=None
    sample_phys_each_epoch = phys_batch_size > 0
    coords_phys = None
    batched_checked = 0
    if sample_phys_each_epoch:
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        t_min, t_max = float('inf'), float('-inf')
        for batch in train_loader:
            if batched_checked > batches_per_epoch * 5:
                break
            x_data, y_data, z_data, t_data, _, _ = [b.to(device) for b in batch]
            x_min, x_max = min(x_min, x_data.min()), max(x_max, x_data.max())
            y_min, y_max = min(y_min, y_data.min()), max(y_max, y_data.max())
            z_min, z_max = min(z_min, z_data.min()), max(z_max, z_data.max())
            t_min, t_max = min(t_min, t_data.min()), max(t_max, t_data.max())
            batched_checked += 1

    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    train_batch_size = train_loader.batch_size
    val_batch_size = val_loader.batch_size

    for epoch in range(n_epochs):
        model.train()  # training mode
        running_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0
        n_batches = 0

        # PDE collocation points (optional re-sample each iteration or epoch)
        if sample_phys_each_epoch:
            x_phys = (x_max - x_min) * torch.rand(phys_batch_size, 1, device=device) + x_min
            y_phys = (y_max - y_min) * torch.rand(phys_batch_size, 1, device=device) + y_min
            z_phys = (z_max - z_min) * torch.rand(phys_batch_size, 1, device=device) + z_min
            t_phys = (t_max - t_min) * torch.rand(phys_batch_size, 1, device=device) + t_min

            coords_phys = (x_phys, y_phys, z_phys, t_phys)

        # 1) Iterate over mini-batches from train_loader
        for batch in train_loader:
            if n_batches > batches_per_epoch:
                break
            # batch is (x, y, z, t, U, V) each shape (batch_size, 1)
            x_data, y_data, z_data, t_data, U_data, V_data = [b.to(device) for b in batch]

            # If we do have coords_phys:
            if coords_phys is not None:
                x_phys, y_phys, z_phys, t_phys = [c.to(device) for c in coords_phys]
                x_phys.requires_grad_(True)
                y_phys.requires_grad_(True)
                z_phys.requires_grad_(True)
                t_phys.requires_grad_(True)
            # else we skip PDE part or sample them here

            optimizer.zero_grad()

            # ----- Data Loss -----
            u_pred_data, v_pred_data = model(x_data, y_data, z_data, t_data)
            data_loss_u = nn.functional.mse_loss(u_pred_data, U_data)
            data_loss_v = nn.functional.mse_loss(v_pred_data, V_data)
            data_loss = data_loss_u  # + data_loss_v

            # ----- PDE Residual Loss (if coords_phys is provided) -----
            if coords_phys is not None:
                res_u, res_v = aliev_panfilov_residual(
                    x_phys, y_phys, z_phys, t_phys, model, params
                )
                pde_loss = (res_u**2).mean() + (res_v**2).mean()
            else:
                pde_loss = torch.tensor(0.0, device=device)

            # Weighted PDE loss
            physics_loss = physics_weight * pde_loss
            loss = data_loss + physics_loss

            loss.backward()
            optimizer.step()

            # Accumulate stats
            running_loss += loss.item()
            running_data_loss += data_loss.item()
            running_pde_loss += pde_loss.item()
            n_batches += 1

        # Averages for the epoch
        epoch_loss = running_loss / n_batches
        epoch_data_loss = running_data_loss / n_batches
        epoch_pde_loss = running_pde_loss / n_batches
        train_loss_hist.append(epoch_loss)
        data_loss_hist.append(epoch_data_loss)
        pde_loss_hist.append(epoch_pde_loss)

        print_string = f"[Epoch {epoch+1}/{n_epochs}] "
        print_string += f"Train Loss: {epoch_loss:.4e} | "
        print_string += f"Data Loss: {epoch_data_loss:.4e} | "
        print_string += f"PDE Loss: {epoch_pde_loss:.4e}"

        if (epoch + 1) % val_every_n_epochs == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            for batch in val_loader:
                if n_val_batches > batches_per_epoch:
                    break
                x_val, y_val, z_val, t_val, U_val, V_val = [b.to(device) for b in batch]
                with torch.no_grad():
                    u_val_pred, v_val_pred = model(x_val, y_val, z_val, t_val)
                    u_loss = nn.functional.mse_loss(u_val_pred, U_val)
                    v_loss = nn.functional.mse_loss(v_val_pred, V_val)
                    data_loss_val = u_loss  # +v_loss
                if sample_phys_each_epoch:
                    x_val_phys = (x_max - x_min) * torch.rand(phys_batch_size, 1, device=device) + x_min
                    y_val_phys = (y_max - y_min) * torch.rand(phys_batch_size, 1, device=device) + y_min
                    z_val_phys = (z_max - z_min) * torch.rand(phys_batch_size, 1, device=device) + z_min
                    t_val_phys = (t_max - t_min) * torch.rand(phys_batch_size, 1, device=device) + t_min
                    x_val_phys, y_val_phys, z_val_phys, t_val_phys = [
                        c.to(device) for c in (x_val_phys, y_val_phys, z_val_phys, t_val_phys)
                    ]
                    x_val_phys.requires_grad_(True)
                    y_val_phys.requires_grad_(True)
                    z_val_phys.requires_grad_(True)
                    t_val_phys.requires_grad_(True)
                    # Compute PDE residual loss with gradients enabled
                    res_u_val, res_v_val = aliev_panfilov_residual(
                        x_val_phys, y_val_phys, z_val_phys, t_val_phys, model, params
                    )
                    pde_loss_val = (res_u_val ** 2).mean() + (res_v_val ** 2).mean()
                else:
                    pde_loss_val = torch.tensor(0.0, device=device)
                total_val_loss = data_loss_val + physics_weight * pde_loss_val
                val_loss += total_val_loss.item()
                n_val_batches += 1

            avg_val_loss = val_loss / n_val_batches
            val_loss_hist.append(avg_val_loss)

            print_string += f" | Val Loss: {avg_val_loss:.4e}"

        # Optional print
        if (epoch + 1) % 1 == 0:
            print(print_string)

    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist
