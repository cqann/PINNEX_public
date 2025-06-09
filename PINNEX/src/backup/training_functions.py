import torch
from torch import nn
import torch.optim as optim

# Use the new eikonal residual function for the physics-based loss
from src.pde.eikonal_pinnex import eikonal_residual
import time
import numpy as np

import torch
from torch.utils.data import DataLoader  # Make sure DataLoader is imported


def sample_surface_points(x_min, x_max, y_min, y_max, z_min, z_max, N, device, is_heart=False):
    # Randomly choose which face (0: x=x_min, 1: x=x_max, ..., 5: z=z_max)
    face_ids = torch.randint(0, 6, (N,), device=device)

    coords = torch.rand(N, 3, device=device)

    # Scale to full range
    coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
    coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min
    coords[:, 2] = coords[:, 2] * (z_max - z_min) + z_min

    # Override one coordinate based on face
    coords[face_ids == 0, 0] = x_min
    coords[face_ids == 1, 0] = x_max
    coords[face_ids == 2, 1] = y_min
    coords[face_ids == 3, 1] = y_max
    coords[face_ids == 4, 2] = z_min
    coords[face_ids == 5, 2] = z_max

    # Return split into x, y, z
    return coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]


def train_pinnex_minibatch_with_ecg_old(wrapper,
                                        train_loader,
                                        val_loader,
                                        params,
                                        n_epochs=10000,
                                        lr=1e-4,
                                        weight_decay=1e-4,
                                        physics_weight=1e-4,
                                        batches_per_epoch=1000,
                                        phys_batch_size=1000,
                                        val_every_n_epochs=5,
                                        device='cpu'):
    """
    Trains a PINN model that uses spatial + ECG inputs.

    train_loader / val_loader must yield:
        (x, y, z, ecg, T, sim_id)
    where:
        x, y, z, T have shape (batch_size, 1)
        ecg has shape (batch_size, n_leads, seq_len)

    The PDE (eikonal) constraint is enforced via randomly sampled collocation points.
    """

    model = wrapper.model
    model = model.to(device)

    for param in wrapper.model.ecg_encoder.parameters():
        if physics_weight > 0:
            param.requires_grad = True
        else:
            param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist = []
    val_loss_hist = []
    data_loss_hist = []
    pde_loss_hist = []

    best_train_loss = float('inf')
    no_improvement_count = 0
    min_lr = 1e-6  # Minimum learning rate for early stopping

    sample_phys_each_epoch = (phys_batch_size > 0) and (physics_weight > 0)
    coords_phys = None

    best_val_loss = float('inf')
    patience = 16
    patience_counter = 0
    best_model_state = None

    is_heart = True

    # Determine spatial domain bounds for PDE collocation from the training set
    if sample_phys_each_epoch:
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        batched_checked = 0

        for batch in train_loader:
            if batched_checked > batches_per_epoch * 5:
                break
            x_data, y_data, z_data, ecg_data, T_data, sim_ids = [b.to(device) for b in batch]
            x_min = min(x_min, x_data.min())
            x_max = max(x_max, x_data.max())
            y_min = min(y_min, y_data.min())
            y_max = max(y_max, y_data.max())
            z_min = min(z_min, z_data.min())
            z_max = max(z_max, z_data.max())
            batched_checked += 1

    for epoch in range(n_epochs):
        wrapper.start_new_epoch()
        model.train()
        running_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0
        n_batches = 0

        # Training batches
        for batch in train_loader:
            if n_batches > batches_per_epoch:
                break
            wrapper.start_new_epoch()

            x_data, y_data, z_data, ecg_data, T_data, sim_ids = [b.to(device) for b in batch]

            optimizer.zero_grad()

            # Forward pass (note: no time input)
            T_pred, c_pred = wrapper(x_data, y_data, z_data, ecg_data, sim_ids=sim_ids)

            # Data loss: mean squared error on activation time T
            data_loss = nn.functional.mse_loss(T_pred, T_data)

            # PDE loss using the eikonal residual: |∇T| - 1/c
            if sample_phys_each_epoch:
                wrapper.start_new_epoch()
                if is_heart:
                    x_phys, y_phys, z_phys = x_data, y_data, z_data
                else:
                    x_phys, y_phys, z_phys = sample_surface_points(
                        x_min, x_max, y_min, y_max, z_min, z_max, phys_batch_size, device)

                x_phys.requires_grad_(True)
                y_phys.requires_grad_(True)
                z_phys.requires_grad_(True)

                # Sample a set of sim_ids and corresponding ECGs from the current batch
                current_batch_size = x_phys.size(0)
                rand_idx = torch.randint(0, sim_ids.shape[0], (1,))
                rand_idx = rand_idx.repeat(current_batch_size)
                phys_sim_ids = sim_ids[rand_idx].to(device)
                ecg_phys = ecg_data[rand_idx]

                residual = eikonal_residual(x_phys, y_phys, z_phys, ecg_phys, wrapper, params, sim_ids=phys_sim_ids)
                pde_loss = (residual**2).mean()
            else:
                pde_loss = torch.tensor(0.0, device=device)

            # Optionally ramp up the physics weight over epochs
            epoch_physics_weight = (epoch / n_epochs) * physics_weight
            loss = data_loss + physics_weight * pde_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_data_loss += data_loss.item()
            running_pde_loss += pde_loss.item()
            n_batches += 1

        epoch_loss = running_loss / n_batches
        epoch_data_loss = running_data_loss / n_batches
        epoch_pde_loss = running_pde_loss / n_batches

        train_loss_hist.append(epoch_loss)
        data_loss_hist.append(epoch_data_loss)
        pde_loss_hist.append(epoch_pde_loss)

        print_string = (f"[Epoch {epoch+1}/{n_epochs}] "
                        f"Train Loss: {epoch_loss:.4e} | "
                        f"Data Loss: {epoch_data_loss:.4e} | "
                        f"PDE Loss: {epoch_pde_loss:.4e}")

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Training loss did not improve for 15 epochs. Reducing learning rate to {current_lr:.4e}")
            no_improvement_count = 0

            if current_lr < min_lr:
                print("Learning rate dropped below minimum. Stopping training early.")
                break

        # Validation
        if (epoch + 1) % val_every_n_epochs == 0:
            wrapper.start_new_epoch()
            model.eval()
            val_loss = 0.0
            re_total = 0.0
            n_val_batches = 0

            for batch in val_loader:
                if n_val_batches > batches_per_epoch:
                    break

                x_val, y_val, z_val, ecg_val, T_val, sim_ids_val = [b.to(device) for b in batch]

                with torch.no_grad():
                    T_val_pred, c_val_pred = wrapper(x_val, y_val, z_val, ecg_val, sim_ids=sim_ids_val)
                    data_loss_val = nn.functional.mse_loss(T_val_pred, T_val)
                    relative_error = torch.norm(T_val_pred - T_val) / torch.norm(T_val)
                    re_total += relative_error.item()

                if sample_phys_each_epoch:

                    if is_heart:
                        x_val_phys, y_val_phys, z_val_phys = x_val, y_val, z_val
                    else:
                        x_val_phys, y_val_phys, z_val_phys = sample_surface_points(
                            x_min, x_max, y_min, y_max, z_min, z_max, phys_batch_size, device)

                    x_val_phys.requires_grad_(True)
                    y_val_phys.requires_grad_(True)
                    z_val_phys.requires_grad_(True)

                    current_batch_size = x_val_phys.size(0)
                    rand_idx = torch.randint(0, sim_ids_val.shape[0], (1,))
                    rand_idx = rand_idx.repeat(current_batch_size)
                    val_phys_sim_ids = sim_ids_val[rand_idx].to(device)
                    ecg_val_phys = ecg_val[rand_idx]

                    residual_val = eikonal_residual(x_val_phys, y_val_phys, z_val_phys,
                                                    ecg_val_phys, wrapper, params, sim_ids=val_phys_sim_ids)
                    pde_loss_val = (residual_val**2).mean()
                else:
                    pde_loss_val = torch.tensor(0.0, device=device)

                total_val_loss = data_loss_val + physics_weight * pde_loss_val
                val_loss += total_val_loss.item()
                n_val_batches += 1

            avg_val_loss = val_loss / n_val_batches
            val_loss_hist.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Validation loss did not improve for {patience} validations. Stopping early.")

                    break
            avg_re = re_total / n_val_batches

            print_string += f" | Val Loss: {avg_val_loss:.4e}"
            print_string += f" | RE: {avg_re:.4e}"

        print(print_string)

    if best_model_state is not None:
        print(f"Loading model with best validation loss: {best_val_loss:.4e}")
        model.load_state_dict(best_model_state)
    else:
        print("No best model state recorded (e.g., validation never ran or no improvement was made). Using the last state of the model.")

    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist


def train_pinnex_minibatch_with_ecg_save(wrapper,
                                         train_loader,
                                         val_loader,
                                         params,
                                         n_epochs=10000,
                                         lr=1e-4,
                                         weight_decay=1e-4,
                                         physics_weight=1e-4,
                                         cv_reg_weight=1e-8,
                                         batches_per_epoch=1000,
                                         phys_batch_size=1000,  # Used when is_heart=False
                                         val_every_n_epochs=5,
                                         device='cpu',
                                         is_heart=True):  # if True, use data points for PDE, else sample
    """
    Trains a PINN model that uses spatial + ECG inputs, with Eikonal PDE loss
    and Conduction Velocity (CV) gradient regularization.

    train_loader / val_loader must yield:
        (x, y, z, ecg, T, sim_id)
    where:
        x, y, z, T have shape (batch_size, 1)
        ecg has shape (batch_size, n_leads, seq_len)
        sim_id has shape (batch_size, 1) or similar if used by wrapper

    The PDE (eikonal) constraint is enforced on collocation points.
    If is_heart=True, data points are used as collocation points.
    If is_heart=False, phys_batch_size points are sampled from the domain.
    CV regularization penalizes the squared magnitude of the gradient of conduction velocity.
    """

    model = wrapper.model
    model = model.to(device)

    for param in wrapper.model.ecg_encoder.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist = []
    val_loss_hist = []
    data_loss_hist = []  # Training data loss
    pde_loss_hist = []  # Training PDE loss
    cv_reg_loss_hist = []  # Training CV regularization loss

    best_train_loss = float('inf')
    no_improvement_count = 0
    min_lr = 1e-6

    # Determine if physics points need to be sampled/prepared for PDE/CV loss
    sample_phys_each_epoch = (physics_weight > 0 or cv_reg_weight > 0)

    best_val_loss = float('inf')
    patience = 16  # For early stopping based on validation loss
    patience_counter = 0
    best_model_state = None

    # Determine spatial domain bounds for PDE collocation if is_heart=False
    x_min, x_max = torch.tensor(float('inf')), torch.tensor(float('-inf'))
    y_min, y_max = torch.tensor(float('inf')), torch.tensor(float('-inf'))
    z_min, z_max = torch.tensor(float('inf')), torch.tensor(float('-inf'))

    if sample_phys_each_epoch and not is_heart:
        print("Determining spatial bounds for physics/CV points (is_heart=False mode)...")
        batched_checked = 0
        for batch_idx, batch_content in enumerate(train_loader):
            if batches_per_epoch > 0 and batched_checked > batches_per_epoch * 5:  # Limit scan
                break
            x_data_cpu, y_data_cpu, z_data_cpu = batch_content[0].cpu(), batch_content[1].cpu(), batch_content[2].cpu()
            x_min = torch.min(x_min, x_data_cpu.min())
            x_max = torch.max(x_max, x_data_cpu.max())
            y_min = torch.min(y_min, y_data_cpu.min())
            y_max = torch.max(y_max, y_data_cpu.max())
            z_min = torch.min(z_min, z_data_cpu.min())
            z_max = torch.max(z_max, z_data_cpu.max())
            batched_checked += 1
        print(
            f"Spatial bounds: X:[{x_min.item():.2f}, {x_max.item():.2f}], Y:[{y_min.item():.2f}, {y_max.item():.2f}], Z:[{z_min.item():.2f}, {z_max.item():.2f}]")

    for epoch in range(n_epochs):
        wrapper.start_new_epoch()  # Assuming this has some model-specific logic
        model.train()
        running_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0
        running_cv_reg_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batches_per_epoch > 0 and n_batches >= batches_per_epoch:
                break

            # Data from loader: x, y, z, ecg_signal, T_target, sim_identifiers
            x_data, y_data, z_data, ecg_batch_data, T_target, sim_ids_batch = [b.to(device) for b in batch]
            current_data_batch_size = x_data.size(0)

            optimizer.zero_grad()

            # --- Data Loss ---
            T_pred_data, _ = wrapper(x_data, y_data, z_data, ecg_batch_data, sim_ids=sim_ids_batch)
            data_loss = nn.functional.mse_loss(T_pred_data, T_target)

            pde_loss_term = torch.tensor(0.0, device=device)
            cv_reg_loss_term = torch.tensor(0.0, device=device)

            if sample_phys_each_epoch:
                # Prepare physics points (x_phys, y_phys, z_phys) and their corresponding ECGs and sim_ids
                if is_heart:
                    # Use the current batch's data points as physics collocation points
                    x_phys = x_data.clone().detach().requires_grad_(True)
                    y_phys = y_data.clone().detach().requires_grad_(True)
                    z_phys = z_data.clone().detach().requires_grad_(True)
                    ecg_for_phys = ecg_batch_data  # Shape: (current_data_batch_size, n_leads, seq_len)
                    sim_ids_for_phys = sim_ids_batch  # Shape: (current_data_batch_size, ...)
                else:
                    # Sample phys_batch_size points randomly from the domain
                    # and assign ECG contexts from the current data batch
                    x_phys_sampled, y_phys_sampled, z_phys_sampled = sample_surface_points(
                        x_min, x_max, y_min, y_max, z_min, z_max, phys_batch_size, device
                    )
                    x_phys = x_phys_sampled.requires_grad_(True)
                    y_phys = y_phys_sampled.requires_grad_(True)
                    z_phys = z_phys_sampled.requires_grad_(True)

                    # Tile/repeat ECGs and sim_ids from current data batch to match phys_batch_size
                    num_repeats = math.ceil(phys_batch_size / current_data_batch_size)
                    ecg_for_phys = ecg_batch_data.repeat_interleave(num_repeats, dim=0)[:phys_batch_size]
                    sim_ids_for_phys = sim_ids_batch.repeat_interleave(num_repeats, dim=0)[:phys_batch_size]

                # --- PDE Loss (Eikonal) ---
                # c_pred_phys will be calculated inside eikonal_residual or by a direct wrapper call
                # The `wrapper` call inside `eikonal_residual` will use (x_phys, y_phys, z_phys, ecg_for_phys, sim_ids_for_phys)
                # to get T_pred_phys and c_pred_phys, ensuring each point uses its own ECG context.
                if physics_weight > 0:
                    # eikonal_residual needs to return T_pred_phys and c_pred_phys if physics_weight > 0
                    # Let's assume eikonal_residual is defined as:
                    # def eikonal_residual(x, y, z, ecg, wrapper, params, sim_ids):
                    #     T_pred, c_pred = wrapper(x, y, z, ecg, sim_ids=sim_ids)
                    #     # ... calculate dT_dx etc. from T_pred ...
                    #     residual_values = torch.sqrt(dT_dx**2 + ...) * c_pred - 1.0
                    #     return residual_values, T_pred, c_pred # Modified to return c_pred
                    # For this example, I'll adapt the original structure where eikonal_residual might only return the residual value,
                    # and c_pred is obtained separately if needed for CV reg when physics_weight is zero.

                    # If eikonal_residual needs c_pred to compute the residual, it should take it or compute it.
                    # Assuming your eikonal_residual is: residual, c_pred_for_pde = func(...)
                    pde_residual_values, c_pred_phys_for_cv = eikonal_residual(
                        x_phys, y_phys, z_phys, ecg_for_phys, wrapper, params, sim_ids=sim_ids_for_phys
                    )
                    pde_loss_term = (pde_residual_values**2).mean()
                elif cv_reg_weight > 0:  # If only CV reg is active, still need c_pred
                    _, c_pred_phys_for_cv = wrapper(x_phys, y_phys, z_phys, ecg_for_phys, sim_ids=sim_ids_for_phys)

                # --- CV Regularization Loss: |∇c|^2 ---
                if cv_reg_weight > 0:
                    if not c_pred_phys_for_cv.requires_grad:
                        # This should not happen if x_phys etc require grad and c_pred_phys_for_cv came from model
                        # Recompute if it was detached or from a non-grad path.
                        # This typically means c_pred_phys_for_cv must be an output of the model with inputs requiring grad.
                        # The c_pred_phys_for_cv from above should be fine.
                        pass  # Or recompute: _, c_pred_phys_for_cv = wrapper(x_phys,...)

                    grad_c = torch.autograd.grad(
                        c_pred_phys_for_cv, [x_phys, y_phys, z_phys],
                        grad_outputs=torch.ones_like(c_pred_phys_for_cv),
                        create_graph=True, allow_unused=True
                    )
                    dc_dx = grad_c[0] if grad_c[0] is not None else torch.zeros_like(x_phys)
                    dc_dy = grad_c[1] if grad_c[1] is not None else torch.zeros_like(y_phys)
                    dc_dz = grad_c[2] if grad_c[2] is not None else torch.zeros_like(z_phys)
                    grad_c_norm_sq = dc_dx**2 + dc_dy**2 + dc_dz**2
                    cv_reg_loss_term = grad_c_norm_sq.mean()

            # --- Total Loss and Backpropagation ---
            loss = data_loss + physics_weight * pde_loss_term + cv_reg_weight * cv_reg_loss_term
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_data_loss += data_loss.item()
            running_pde_loss += pde_loss_term.item()  # Use .item() for scalar tensor
            running_cv_reg_loss += cv_reg_loss_term.item()  # Use .item()
            n_batches += 1

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader) if batches_per_epoch <=0 else batches_per_epoch}: "
                      f"Cur Batch Loss: {loss.item():.4e}, Data: {data_loss.item():.4e}, "
                      f"PDE: {pde_loss_term.item():.4e}, CV-Reg: {cv_reg_loss_term.item():.4e}")

        epoch_loss = running_loss / n_batches if n_batches > 0 else 0
        epoch_data_loss = running_data_loss / n_batches if n_batches > 0 else 0
        epoch_pde_loss = running_pde_loss / n_batches if n_batches > 0 else 0
        epoch_cv_reg_loss = running_cv_reg_loss / n_batches if n_batches > 0 else 0

        train_loss_hist.append(epoch_loss)
        data_loss_hist.append(epoch_data_loss)
        pde_loss_hist.append(epoch_pde_loss)
        cv_reg_loss_hist.append(epoch_cv_reg_loss)

        print_string = (f"[Epoch {epoch+1}/{n_epochs}] Train Loss: {epoch_loss:.4e} | "
                        f"Data: {epoch_data_loss:.4e} | PDE: {epoch_pde_loss:.4e} | "
                        f"CV-Reg: {epoch_cv_reg_loss:.4e}")

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 5:  # Adjust as needed
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] / 5, min_lr)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train loss stagnated. Reducing LR to {current_lr:.4e}")
            no_improvement_count = 0
            if abs(current_lr - min_lr) < 1e-9 or current_lr < min_lr:
                print("LR at minimum. Stopping early.")
                break

        # --- Validation ---
        if (epoch + 1) % val_every_n_epochs == 0:
            model.eval()
            val_loss_epoch = 0.0
            val_data_loss_epoch = 0.0
            val_pde_loss_epoch = 0.0
            val_cv_reg_loss_epoch = 0.0
            re_total = 0.0
            n_val_batches = 0

            with torch.no_grad():  # Overall no_grad for validation data predictions
                for val_batch_idx, batch_val in enumerate(val_loader):
                    # Cap val batches similar to train if batches_per_epoch is set
                    if batches_per_epoch > 0 and val_batch_idx >= batches_per_epoch:
                        break

                    x_val, y_val, z_val, ecg_val_batch, T_val_target, sim_ids_val_batch = [
                        b.to(device) for b in batch_val]
                    current_val_batch_size = x_val.size(0)

                    T_val_pred, _ = wrapper(x_val, y_val, z_val, ecg_val_batch, sim_ids=sim_ids_val_batch)
                    val_data_loss_item = nn.functional.mse_loss(T_val_pred, T_val_target)

                    relative_error = torch.norm(T_val_pred - T_val_target) / (torch.norm(T_val_target) + 1e-8)
                    re_total += relative_error.item()

                    val_pde_loss_item = torch.tensor(0.0, device=device)
                    val_cv_reg_loss_item = torch.tensor(0.0, device=device)

                    if sample_phys_each_epoch:
                        # Prepare physics points for validation
                        # Gradients needed temporarily for internal autograd of PDE/CV loss terms
                        # So, enable grad locally for these specific computations.
                        with torch.enable_grad():
                            if is_heart:
                                x_val_phys = x_val.clone().detach().requires_grad_(True)
                                y_val_phys = y_val.clone().detach().requires_grad_(True)
                                z_val_phys = z_val.clone().detach().requires_grad_(True)
                                ecg_for_val_phys = ecg_val_batch
                                sim_ids_for_val_phys = sim_ids_val_batch
                            else:
                                x_val_phys_sampled, y_val_phys_sampled, z_val_phys_sampled = sample_surface_points(
                                    x_min, x_max, y_min, y_max, z_min, z_max, phys_batch_size, device
                                )
                                x_val_phys = x_val_phys_sampled.requires_grad_(True)
                                y_val_phys = y_val_phys_sampled.requires_grad_(True)
                                z_val_phys = z_val_phys_sampled.requires_grad_(True)

                                num_repeats_val = math.ceil(phys_batch_size / current_val_batch_size)
                                ecg_for_val_phys = ecg_val_batch.repeat_interleave(
                                    num_repeats_val, dim=0)[:phys_batch_size]
                                sim_ids_for_val_phys = sim_ids_val_batch.repeat_interleave(num_repeats_val, dim=0)[
                                    :phys_batch_size]

                            # --- PDE Loss (Validation) ---
                            c_val_pred_phys_for_cv = None  # Initialize
                            if physics_weight > 0:
                                val_pde_residuals, c_val_pred_phys_for_cv = eikonal_residual(
                                    x_val_phys, y_val_phys, z_val_phys, ecg_for_val_phys, wrapper, params, sim_ids=sim_ids_for_val_phys
                                )
                                val_pde_loss_item = (val_pde_residuals**2).mean()
                            elif cv_reg_weight > 0:  # Still need c_val_pred if only CV reg active
                                _, c_val_pred_phys_for_cv = wrapper(
                                    x_val_phys, y_val_phys, z_val_phys, ecg_for_val_phys, sim_ids=sim_ids_for_val_phys)

                            # --- CV Reg Loss (Validation) ---
                            if cv_reg_weight > 0:
                                grad_c_val = torch.autograd.grad(
                                    c_val_pred_phys_for_cv, [x_val_phys, y_val_phys, z_val_phys],
                                    grad_outputs=torch.ones_like(c_val_pred_phys_for_cv),
                                    create_graph=False,  # No graph needed for validation's own backward pass
                                    allow_unused=True
                                )
                                dc_dx_val = grad_c_val[0] if grad_c_val[0] is not None else torch.zeros_like(x_val_phys)
                                dc_dy_val = grad_c_val[1] if grad_c_val[1] is not None else torch.zeros_like(y_val_phys)
                                dc_dz_val = grad_c_val[2] if grad_c_val[2] is not None else torch.zeros_like(z_val_phys)
                                grad_c_norm_sq_val = dc_dx_val**2 + dc_dy_val**2 + dc_dz_val**2
                                val_cv_reg_loss_item = grad_c_norm_sq_val.mean()

                        # Detach losses as they are for metrics only and computed within enable_grad
                        val_pde_loss_item = val_pde_loss_item.detach()
                        val_cv_reg_loss_item = val_cv_reg_loss_item.detach()

                    total_val_loss_components = val_data_loss_item + physics_weight * val_pde_loss_item + cv_reg_weight * val_cv_reg_loss_item

                    val_loss_epoch += total_val_loss_components.item()
                    val_data_loss_epoch += val_data_loss_item.item()
                    val_pde_loss_epoch += val_pde_loss_item.item()
                    val_cv_reg_loss_epoch += val_cv_reg_loss_item.item()
                    n_val_batches += 1

            avg_val_loss = val_loss_epoch / n_val_batches if n_val_batches > 0 else 0
            avg_val_data_loss = val_data_loss_epoch / n_val_batches if n_val_batches > 0 else 0
            avg_val_pde_loss = val_pde_loss_epoch / n_val_batches if n_val_batches > 0 else 0
            avg_val_cv_reg_loss = val_cv_reg_loss_epoch / n_val_batches if n_val_batches > 0 else 0
            val_loss_hist.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                print(f"*** New best validation loss: {best_val_loss:.4e} ***")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Validation loss did not improve for {patience} validations. Stopping early.")
                    if best_model_state:
                        model.load_state_dict(best_model_state)  # Load best model before stopping
                    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist, cv_reg_loss_hist

            avg_re = re_total / n_val_batches if n_val_batches > 0 else 0
            print_string += (f" | Val Loss: {avg_val_loss:.4e} (Data: {avg_val_data_loss:.4e}, "
                             f"PDE: {avg_val_pde_loss:.4e}, CV-Reg: {avg_val_cv_reg_loss:.4e}) | RE: {avg_re:.4e}")
        print(print_string)

    if best_model_state is not None:
        print(f"Loading model with best validation loss: {best_val_loss:.4e}")
        model.load_state_dict(best_model_state)
    else:
        print("No best model state recorded (or early stopping before first validation). Using the last state.")

    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist  # , cv_reg_loss_hist


def train_pinnex_minibatch_with_ecg(wrapper,
                                    train_loader,
                                    val_loader,
                                    params,  # General parameters, might be used by eikonal_residual or wrapper
                                    n_epochs=10000,
                                    lr=1e-4,
                                    weight_decay=1e-4,
                                    physics_weight=1e-4,
                                    cv_reg_weight=1e-8,
                                    batches_per_epoch=1000,  # Max batches per epoch from train_loader
                                    phys_batch_size=1000,  # Num points to sample for physics loss
                                    val_every_n_epochs=5,
                                    device='cpu',
                                    collocation_points_filepath="coll_points.dat",  # Path to .dat file
                                    is_heart=True):  # If True, uses collocation_points_filepath
    """
    Trains a PINN model that uses spatial + ECG inputs, with Eikonal PDE loss
    and Conduction Velocity (CV) gradient regularization.

    If is_heart=True, physics points are sampled from 'collocation_points_filepath'.
    Otherwise (is_heart=False), they are sampled from the bounding box of training data.
    """

    model = wrapper.model
    model = model.to(device)

    # This logic for ecg_encoder parameters seems to always set requires_grad = True
    for param in wrapper.model.ecg_encoder.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist, val_loss_hist = [], []
    data_loss_hist, pde_loss_hist, cv_reg_loss_hist = [], [], []  # For epoch averages

    best_train_loss = float('inf')
    no_improvement_count = 0  # For LR scheduler based on train loss
    min_lr = 1e-7  # Adjusted minimum learning rate

    sample_phys_each_epoch = (phys_batch_size > 0) and (physics_weight > 0 or cv_reg_weight > 0)

    heart_collocation_points = None
    if is_heart and sample_phys_each_epoch:
        try:
            print(f"Loading collocation points from: {collocation_points_filepath}")
            loaded_points = np.loadtxt(collocation_points_filepath)  # Assumes space-separated x y z
            if loaded_points.ndim == 1 and loaded_points.shape[0] == 3:  # Handle single point case
                loaded_points = loaded_points.reshape(1, 3)
            elif loaded_points.ndim != 2 or loaded_points.shape[1] != 3:
                raise ValueError(
                    f"Collocation file {collocation_points_filepath} must have 3 columns (X,Y,Z). Got shape {loaded_points.shape}")
            heart_collocation_points = torch.from_numpy(loaded_points).float().to(device)
            print(f"Loaded {heart_collocation_points.shape[0]} collocation points for 'is_heart=True' mode.")
            if heart_collocation_points.shape[0] == 0:
                print("Warning: Loaded collocation points file is empty.")
                heart_collocation_points = None  # Treat as no points loaded
        except Exception as e:
            print(f"Error loading collocation points from {collocation_points_filepath}: {e}")
            print("CRITICAL: Proceeding without pre-loaded collocation points for is_heart=True. Physics loss may not be calculated.")
            heart_collocation_points = None  # Ensure it's None if loading failed

    # Determine spatial domain bounds if sampling from bounding box (is_heart=False)
    x_min_bound, x_max_bound = torch.tensor(float('inf'), device=device), torch.tensor(float('-inf'), device=device)
    y_min_bound, y_max_bound = torch.tensor(float('inf'), device=device), torch.tensor(float('-inf'), device=device)
    z_min_bound, z_max_bound = torch.tensor(float('inf'), device=device), torch.tensor(float('-inf'), device=device)

    if sample_phys_each_epoch and not is_heart:
        print("Determining spatial bounds for physics/CV points (bounding box sampling for is_heart=False)...")
        batched_checked = 0
        # Limit batches checked for bounds to avoid iterating over entire dataset if very large
        # batches_per_epoch_bounds = batches_per_epoch if batches_per_epoch > 0 else len(train_loader)
        # limit_batches_for_bounds = min(batches_per_epoch_bounds * 5, len(train_loader))

        temp_loader_iter = iter(train_loader)
        # Check a few batches to get an idea of the bounds
        # Adjust num_batches_for_bounds as needed, e.g., 50 or up to a certain fraction of train_loader
        num_batches_for_bounds = min(50, len(train_loader))
        print(f"Checking up to {num_batches_for_bounds} batches for spatial bounds.")
        for i in range(num_batches_for_bounds):
            try:
                batch_content = next(temp_loader_iter)
                x_data_b, y_data_b, z_data_b = batch_content[0].to(
                    device), batch_content[1].to(device), batch_content[2].to(device)
                x_min_bound, x_max_bound = torch.min(x_min_bound, x_data_b.min()
                                                     ), torch.max(x_max_bound, x_data_b.max())
                y_min_bound, y_max_bound = torch.min(y_min_bound, y_data_b.min()
                                                     ), torch.max(y_max_bound, y_data_b.max())
                z_min_bound, z_max_bound = torch.min(z_min_bound, z_data_b.min()
                                                     ), torch.max(z_max_bound, z_data_b.max())
            except StopIteration:
                break  # Exhausted train_loader
        del temp_loader_iter
        print(
            f"Bounding Box Spatial bounds: X:[{x_min_bound.item():.2f}, {x_max_bound.item():.2f}], Y:[{y_min_bound.item():.2f}, {y_max_bound.item():.2f}], Z:[{z_min_bound.item():.2f}, {z_max_bound.item():.2f}]")
        if torch.isinf(x_min_bound):  # Check if bounds were actually updated
            print("Warning: Bounding box for physics points could not be determined. Physics loss for is_heart=False might be problematic.")

    best_val_loss = float('inf')
    patience = 16  # Original patience for early stopping
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        wrapper.start_new_epoch()  # If your wrapper needs this
        model.train()
        running_loss_epoch, running_data_loss_epoch, running_pde_loss_epoch, running_cv_reg_loss_epoch = 0.0, 0.0, 0.0, 0.0
        n_batches_processed = 0

        for batch_idx, batch_content in enumerate(train_loader):
            if batches_per_epoch > 0 and n_batches_processed >= batches_per_epoch:
                break

            x_data, y_data, z_data, ecg_data, T_data, sim_ids = [b.to(device) for b in batch_content]
            current_data_batch_size = x_data.size(0)

            optimizer.zero_grad()

            # --- Data Loss ---
            T_pred_data, _ = wrapper(x_data, y_data, z_data, ecg_data, sim_ids=sim_ids)
            data_loss = nn.functional.mse_loss(T_pred_data, T_data)

            # --- Physics Loss (PDE & CV Regularization) ---
            pde_loss_batch = torch.tensor(0.0, device=device)
            cv_reg_loss_batch = torch.tensor(0.0, device=device)

            if sample_phys_each_epoch and current_data_batch_size > 0:  # Need data ECGs for context
                x_phys, y_phys, z_phys = None, None, None
                ecg_for_physics, sim_ids_for_physics = None, None
                actual_phys_batch_size = 0  # Number of physics points we actually process

                if is_heart:
                    if heart_collocation_points is not None and heart_collocation_points.shape[0] > 0:
                        num_total_coll_points = heart_collocation_points.shape[0]
                        actual_phys_batch_size = min(
                            phys_batch_size, num_total_coll_points) if phys_batch_size > num_total_coll_points else phys_batch_size

                        random_indices = torch.randint(0, num_total_coll_points,
                                                       (actual_phys_batch_size,), device=device)
                        selected_points = heart_collocation_points[random_indices]

                        x_phys = selected_points[:, 0:1].detach().clone().requires_grad_(True)
                        y_phys = selected_points[:, 1:2].detach().clone().requires_grad_(True)
                        z_phys = selected_points[:, 2:3].detach().clone().requires_grad_(True)
                else:  # Sample from domain bounding box (is_heart=False)
                    if not torch.isinf(x_min_bound):  # Check if bounds are valid
                        actual_phys_batch_size = phys_batch_size
                        x_phys, y_phys, z_phys = sample_surface_points(
                            x_min_bound, x_max_bound, y_min_bound, y_max_bound, z_min_bound, z_max_bound,
                            actual_phys_batch_size, device
                        )
                        x_phys.requires_grad_(True)
                        y_phys.requires_grad_(True)
                        z_phys.requires_grad_(True)

                if x_phys is not None and actual_phys_batch_size > 0:  # If physics points were successfully sampled
                    # Assign ECG context from the current data batch
                    ecg_select_indices = torch.randint(0, current_data_batch_size,
                                                       (actual_phys_batch_size,), device=device)
                    random_index = torch.randint(0, current_data_batch_size, (1,), device=device)
                    ecg_select_indices = random_index.repeat(actual_phys_batch_size)
                    ecg_for_physics = ecg_data[ecg_select_indices]
                    sim_ids_for_physics = sim_ids[ecg_select_indices]

                    # Calculate PDE and CV losses only if we have valid physics points and context
                    c_pred_phys_for_cv_reg = None
                    if physics_weight > 0:
                        residual, c_pred_phys = eikonal_residual(
                            x_phys, y_phys, z_phys, ecg_for_physics, wrapper, params, sim_ids=sim_ids_for_physics)
                        pde_loss_batch = (residual**2).mean()
                        c_pred_phys_for_cv_reg = c_pred_phys

                    if cv_reg_weight > 0:
                        if c_pred_phys_for_cv_reg is None:  # If only CV reg is active or PDE part failed
                            _, c_pred_phys_for_cv_reg = wrapper(
                                x_phys, y_phys, z_phys, ecg_for_physics, sim_ids=sim_ids_for_physics)

                        if c_pred_phys_for_cv_reg is not None:
                            grad_c = torch.autograd.grad(
                                c_pred_phys_for_cv_reg, [x_phys, y_phys, z_phys],
                                grad_outputs=torch.ones_like(c_pred_phys_for_cv_reg),
                                create_graph=True, allow_unused=True
                            )
                            dc_dx = grad_c[0] if grad_c[0] is not None else torch.zeros_like(x_phys)
                            dc_dy = grad_c[1] if grad_c[1] is not None else torch.zeros_like(y_phys)
                            dc_dz = grad_c[2] if grad_c[2] is not None else torch.zeros_like(z_phys)
                            grad_c_norm_sq = dc_dx**2 + dc_dy**2 + dc_dz**2
                            cv_reg_loss_batch = grad_c_norm_sq.mean()

            # --- Total Loss and Optimization Step ---
            loss = data_loss + physics_weight * pde_loss_batch + cv_reg_weight * cv_reg_loss_batch
            loss.backward()
            optimizer.step()

            running_loss_epoch += loss.item()
            running_data_loss_epoch += data_loss.item()
            running_pde_loss_epoch += pde_loss_batch.item()
            running_cv_reg_loss_epoch += cv_reg_loss_batch.item()
            n_batches_processed += 1

            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader) if batches_per_epoch <=0 else batches_per_epoch}: "
                      f"Current Batch Loss: {loss.item():.4e}, Data: {data_loss.item():.4e}, "
                      f"PDE: {pde_loss_batch.item():.4e}, CV-Reg: {cv_reg_loss_batch.item():.4e}")

        # --- Epoch Summary ---
        avg_epoch_loss = running_loss_epoch / n_batches_processed if n_batches_processed > 0 else 0
        avg_epoch_data_loss = running_data_loss_epoch / n_batches_processed if n_batches_processed > 0 else 0
        avg_epoch_pde_loss = running_pde_loss_epoch / n_batches_processed if n_batches_processed > 0 else 0
        avg_epoch_cv_reg_loss = running_cv_reg_loss_epoch / n_batches_processed if n_batches_processed > 0 else 0

        train_loss_hist.append(avg_epoch_loss)
        data_loss_hist.append(avg_epoch_data_loss)
        pde_loss_hist.append(avg_epoch_pde_loss)
        cv_reg_loss_hist.append(avg_epoch_cv_reg_loss)

        print_string = (f"[Epoch {epoch+1}/{n_epochs}] Train Loss: {avg_epoch_loss:.4e} | "
                        f"Data: {avg_epoch_data_loss:.4e} | PDE: {avg_epoch_pde_loss:.4e} | "
                        f"CV-Reg: {avg_epoch_cv_reg_loss:.4e}")

        # --- Learning Rate Scheduler (based on training loss stagnation) ---
        if avg_epoch_loss < best_train_loss:  # Using average epoch loss
            best_train_loss = avg_epoch_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 5:  # Reduce LR if no improvement for 5 epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] / 5, min_lr)  # Reduce by factor of 5
            current_lr_val = optimizer.param_groups[0]['lr']
            print(f"Training loss stagnated. Reducing learning rate to {current_lr_val:.4e}")
            no_improvement_count = 0
            if abs(current_lr_val - min_lr) < 1e-9 or current_lr_val < min_lr:
                print("Learning rate reached minimum. Stopping training early.")
                break

        # --- Validation Step ---
        if (epoch + 1) % val_every_n_epochs == 0:
            model.eval()
            running_val_loss_epoch, running_val_data_loss_epoch = 0.0, 0.0
            running_val_pde_loss_epoch, running_val_cv_reg_loss_epoch = 0.0, 0.0
            running_re_total = 0.0
            n_val_batches_processed = 0

            with torch.no_grad():
                for val_batch_idx, batch_val_content in enumerate(val_loader):
                    # Cap validation batches similar to training if batches_per_epoch is set
                    if batches_per_epoch > 0 and n_val_batches_processed >= batches_per_epoch:
                        break

                    x_val, y_val, z_val, ecg_val, T_val, sim_ids_val = [b.to(device) for b in batch_val_content]
                    current_val_batch_size = x_val.size(0)

                    T_val_pred, _ = wrapper(x_val, y_val, z_val, ecg_val, sim_ids=sim_ids_val)
                    val_data_loss = nn.functional.mse_loss(T_val_pred, T_val)

                    relative_error = torch.norm(T_val_pred.squeeze() - T_val.squeeze()) / \
                        (torch.norm(T_val.squeeze()) + 1e-8)
                    running_re_total += relative_error.item()

                    val_pde_loss_batch = torch.tensor(0.0, device=device)
                    val_cv_reg_loss_batch = torch.tensor(0.0, device=device)

                    if sample_phys_each_epoch and current_val_batch_size > 0:
                        x_val_phys, y_val_phys, z_val_phys = None, None, None
                        ecg_val_phys, sim_ids_val_phys = None, None
                        actual_val_phys_batch_size = 0

                        if is_heart:
                            if heart_collocation_points is not None and heart_collocation_points.shape[0] > 0:
                                num_total_coll_points_val = heart_collocation_points.shape[0]
                                actual_val_phys_batch_size = min(
                                    phys_batch_size, num_total_coll_points_val) if phys_batch_size > num_total_coll_points_val else phys_batch_size

                                random_indices_val = torch.randint(
                                    0, num_total_coll_points_val, (actual_val_phys_batch_size,), device=device)
                                selected_points_val = heart_collocation_points[random_indices_val]

                                x_val_phys = selected_points_val[:, 0:1].detach().clone().requires_grad_(True)
                                y_val_phys = selected_points_val[:, 1:2].detach().clone().requires_grad_(True)
                                z_val_phys = selected_points_val[:, 2:3].detach().clone().requires_grad_(True)
                        else:  # Sample from bounding box for validation
                            if not torch.isinf(x_min_bound):
                                actual_val_phys_batch_size = phys_batch_size
                                x_val_phys, y_val_phys, z_val_phys = sample_surface_points(
                                    x_min_bound, x_max_bound, y_min_bound, y_max_bound, z_min_bound, z_max_bound,
                                    actual_val_phys_batch_size, device
                                )
                                x_val_phys.requires_grad_(True)
                                y_val_phys.requires_grad_(True)
                                z_val_phys.requires_grad_(True)

                        if x_val_phys is not None and actual_val_phys_batch_size > 0:
                            ecg_select_indices_val = torch.randint(
                                0, current_val_batch_size, (actual_val_phys_batch_size,), device=device)
                            random_index = torch.randint(0, current_val_batch_size, (1,), device=device)
                            ecg_select_indices_val = random_index.repeat(actual_val_phys_batch_size)
                            ecg_val_phys = ecg_val[ecg_select_indices_val]
                            sim_ids_val_phys = sim_ids_val[ecg_select_indices_val]

                            with torch.enable_grad():  # Gradients needed for autograd inside eikonal_residual_fn
                                c_val_pred_phys_for_cv_reg = None
                                if physics_weight > 0:
                                    residual_val, c_val_pred_phys = eikonal_residual(
                                        x_val_phys, y_val_phys, z_val_phys,
                                        ecg_val_phys, wrapper, params, sim_ids=sim_ids_val_phys)
                                    val_pde_loss_batch = (residual_val**2).mean()
                                    c_val_pred_phys_for_cv_reg = c_val_pred_phys

                                if cv_reg_weight > 0:
                                    if c_val_pred_phys_for_cv_reg is None:
                                        _, c_val_pred_phys_for_cv_reg = wrapper(
                                            x_val_phys, y_val_phys, z_val_phys, ecg_val_phys, sim_ids=sim_ids_val_phys)

                                    if c_val_pred_phys_for_cv_reg is not None:
                                        grad_c_val = torch.autograd.grad(
                                            c_val_pred_phys_for_cv_reg, [x_val_phys, y_val_phys, z_val_phys],
                                            grad_outputs=torch.ones_like(c_val_pred_phys_for_cv_reg),
                                            create_graph=False, allow_unused=True)  # create_graph=False for validation
                                        dc_dx_val = grad_c_val[0] if grad_c_val[0] is not None else torch.zeros_like(
                                            x_val_phys)
                                        dc_dy_val = grad_c_val[1] if grad_c_val[1] is not None else torch.zeros_like(
                                            y_val_phys)
                                        dc_dz_val = grad_c_val[2] if grad_c_val[2] is not None else torch.zeros_like(
                                            z_val_phys)
                                        grad_c_norm_sq_val = dc_dx_val**2 + dc_dy_val**2 + dc_dz_val**2
                                        val_cv_reg_loss_batch = grad_c_norm_sq_val.mean()

                            val_pde_loss_batch = val_pde_loss_batch.detach()
                            val_cv_reg_loss_batch = val_cv_reg_loss_batch.detach()

                    current_val_total_loss = val_data_loss + physics_weight * val_pde_loss_batch + cv_reg_weight * val_cv_reg_loss_batch
                    running_val_loss_epoch += current_val_total_loss.item()
                    running_val_data_loss_epoch += val_data_loss.item()
                    running_val_pde_loss_epoch += val_pde_loss_batch.item()
                    running_val_cv_reg_loss_epoch += val_cv_reg_loss_batch.item()
                    n_val_batches_processed += 1

            avg_val_loss = running_val_loss_epoch / n_val_batches_processed if n_val_batches_processed > 0 else 0
            avg_val_data_loss = running_val_data_loss_epoch / n_val_batches_processed if n_val_batches_processed > 0 else 0
            avg_val_pde_loss = running_val_pde_loss_epoch / n_val_batches_processed if n_val_batches_processed > 0 else 0
            avg_val_cv_reg_loss = running_val_cv_reg_loss_epoch / n_val_batches_processed if n_val_batches_processed > 0 else 0
            avg_re = running_re_total / n_val_batches_processed if n_val_batches_processed > 0 else 0
            val_loss_hist.append(avg_val_loss)

            print_string += (f" | Val Loss: {avg_val_loss:.4e} (Data: {avg_val_data_loss:.4e}, "
                             f"PDE: {avg_val_pde_loss:.4e}, CV-Reg: {avg_val_cv_reg_loss:.4e}) | RE: {avg_re:.4e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()  # Save best model state
                print(f"*** New best validation loss: {best_val_loss:.4e} ***")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Validation loss did not improve for {patience} validation steps. Stopping early.")
                    # Load the best model before exiting
                    if best_model_state is not None:
                        print(f"Loading model with best validation loss: {best_val_loss:.4e}")
                        model.load_state_dict(best_model_state)
                    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist  # , cv_reg_loss_hist

        print(print_string)
        # Check early stopping from LR scheduler as well
        if optimizer.param_groups[0]['lr'] <= min_lr and no_improvement_count == 0:  # Check if LR hit min and was just reset
            if abs(optimizer.param_groups[0]['lr'] - min_lr) < 1e-9:  # Double check due to float precision
                print("Learning rate at minimum after reduction. Considering early stop post validation.")
                # This break is already handled by the LR scheduler logic above if it's triggered by training loss.
                # The validation patience break is separate.

    # End of epochs loop
    if best_model_state is not None:
        print(f"Training finished. Loading model with best validation loss: {best_val_loss:.4e}")
        model.load_state_dict(best_model_state)
    else:
        print("Training finished. No best model state recorded (e.g., if validation was not run or did not improve). Using last model state.")

    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist  # , cv_reg_loss_hist


def train_lbfgs_with_ecg(
    wrapper,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1,
    device='cpu',
    # --------------------- ADDED
    params=None,            # dictionary for PDE parameters
    physics_weight=1e-4,
    phys_batch_size=1000
    # --------------------- end ADDED
):
    """
    Modified version that includes PDE (eikonal) loss.
    """
    torch.manual_seed(42)

    # --------------------- ADDED: figure out bounding box for sampling PDE collocation
    if physics_weight > 0 and phys_batch_size > 0:
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')

        # One pass through training data to get bounds
        for batch in train_loader:
            x_data, y_data, z_data, ecg_data, T_data, sim_ids = batch
            x_min = min(x_min, x_data.min().item())
            x_max = max(x_max, x_data.max().item())
            y_min = min(y_min, y_data.min().item())
            y_max = max(y_max, y_data.max().item())
            z_min = min(z_min, z_data.min().item())
            z_max = max(z_max, z_data.max().item())
    # --------------------- end ADDED

    model = wrapper.model.to(device)
    model.train()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    train_loss_hist = []
    val_loss_hist = []
    data_loss_hist = []
    pde_loss_hist = []

    import copy
    import math

    # Within your train_lbfgs_with_ecg function, replace the epoch loop with the following:

    for epoch in range(n_epochs):
        restart_epoch = False
        while True:
            # Save a backup of the model state before starting the epoch
            backup_model = copy.deepcopy(model.state_dict())

            epoch_loss = 0.0
            data_loss_epoch = 0.0
            pde_loss_epoch = 0.0
            n_batches = 0

            # --------------------- (Unchanged) PDE collocation sampling and inner loop setup remain here
            if physics_weight > 0 and phys_batch_size > 0:
                x_phys = (x_max - x_min) * torch.rand(phys_batch_size, 1, device=device) + x_min
                y_phys = (y_max - y_min) * torch.rand(phys_batch_size, 1, device=device) + y_min
                z_phys = (z_max - z_min) * torch.rand(phys_batch_size, 1, device=device) + z_min

                x_phys.requires_grad_(True)
                y_phys.requires_grad_(True)
                z_phys.requires_grad_(True)
            else:
                x_phys = y_phys = z_phys = None
            # --------------------- end unchanged

            for batch in train_loader:
                x, y, z, ecg, T_data, sim_ids = [b.to(device) for b in batch]

                def closure():
                    optimizer.zero_grad()
                    model.eval()
                    T_pred, _ = wrapper(x, y, z, ecg, sim_ids=sim_ids)
                    data_loss = torch.nn.functional.mse_loss(T_pred, T_data)

                    if (physics_weight > 0) and (x_phys is not None):
                        rand_idx = torch.randint(0, sim_ids.shape[0], (1,))
                        rand_idx = rand_idx.repeat(phys_batch_size)
                        phys_sim_ids = sim_ids[rand_idx]
                        ecg_phys = ecg[rand_idx]

                        residual = eikonal_residual(x_phys, y_phys, z_phys,
                                                    ecg_phys, wrapper,
                                                    params, sim_ids=phys_sim_ids)
                        pde_loss = (residual**2).mean()
                    else:
                        pde_loss = torch.tensor(0.0, device=device)

                    loss = data_loss + physics_weight * pde_loss
                    loss.backward()

                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-1e2, 1e2)
                    return loss

                optimizer.step(closure)

                with torch.enable_grad():
                    T_pred, _ = wrapper(x, y, z, ecg, sim_ids=sim_ids)
                    data_loss = torch.nn.functional.mse_loss(T_pred, T_data)

                    if (physics_weight > 0) and (x_phys is not None):
                        rand_idx = torch.randint(0, sim_ids.shape[0], (1,))
                        rand_idx = rand_idx.repeat(phys_batch_size)
                        phys_sim_ids = sim_ids[rand_idx]
                        ecg_phys = ecg[rand_idx]

                        residual = eikonal_residual(x_phys, y_phys, z_phys,
                                                    ecg_phys, wrapper,
                                                    params, sim_ids=phys_sim_ids)
                        pde_loss = (residual**2).mean()
                    else:
                        pde_loss = torch.tensor(0.0, device=device)

                    loss_val = data_loss + physics_weight * pde_loss

                epoch_loss += loss_val.item()
                data_loss_epoch += data_loss.item()
                pde_loss_epoch += pde_loss.item()
                n_batches += 1
                # Check for NaN training loss; if found, revert backup and half the learning rate before re-running the epoch.
                if math.isnan(epoch_loss):
                    restart_epoch = True
                    break

            avg_train_loss = epoch_loss / n_batches
            avg_data_loss = data_loss_epoch / n_batches
            avg_pde_loss = pde_loss_epoch / n_batches

            if restart_epoch:
                print(f"Epoch {epoch+1}: NaN encountered. Halving learning rate and reverting model backup.")
                model.load_state_dict(backup_model)
                for group in optimizer.param_groups:
                    group['lr'] /= 2
                # Restart the epoch with updated learning rate.
                break

            # If no NaN, update the history trackers and proceed.
            train_loss_hist.append(avg_train_loss)
            data_loss_hist.append(avg_data_loss)
            pde_loss_hist.append(avg_pde_loss)

            # (Unchanged) Validation loop and printing occur here.
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, z, ecg, T_data, sim_ids = [b.to(device) for b in batch]
                    T_pred, _ = wrapper(x, y, z, ecg, sim_ids=sim_ids)
                    loss_val = torch.nn.functional.mse_loss(T_pred, T_data)
                    val_loss += loss_val.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else 0
            val_loss_hist.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Train Loss = {avg_train_loss:.4e}, "
                  f"Val Loss = {avg_val_loss:.4e}")
            model.train()

            break  # Exit the while loop and move to the next epoch

    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist
