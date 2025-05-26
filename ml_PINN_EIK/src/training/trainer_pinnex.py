import torch
from torch import nn
import torch.optim as optim

# Use the new eikonal residual function for the physics-based loss
from src.pde.eikonal_pinnex import eikonal_residual


def train_pinnex_minibatch_with_ecg(wrapper,
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
    patience = 50
    patience_counter = 0
    best_model_state = None

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

        # Resample PDE collocation points (spatial only)
        if sample_phys_each_epoch:
            x_phys = (x_max - x_min) * torch.rand(phys_batch_size, 1, device=device) + x_min
            y_phys = (y_max - y_min) * torch.rand(phys_batch_size, 1, device=device) + y_min
            z_phys = (z_max - z_min) * torch.rand(phys_batch_size, 1, device=device) + z_min
            coords_phys = (x_phys, y_phys, z_phys)

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
            if coords_phys is not None:
                wrapper.start_new_epoch()
                x_phys, y_phys, z_phys = coords_phys
                x_phys.requires_grad_(True)
                y_phys.requires_grad_(True)
                z_phys.requires_grad_(True)

                # Sample a set of sim_ids and corresponding ECGs from the current batch
                rand_idx = torch.randint(0, sim_ids.shape[0], (1,))
                rand_idx = rand_idx.repeat(phys_batch_size)
                phys_sim_ids = sim_ids[rand_idx].to(device)
                ecg_phys = ecg_data[rand_idx]

                residual = eikonal_residual(x_phys, y_phys, z_phys, ecg_phys, wrapper, params, sim_ids=phys_sim_ids)
                pde_loss = (residual**2).mean()
            else:
                pde_loss = torch.tensor(0.0, device=device)

            # Optionally ramp up the physics weight over epochs
            epoch_physics_weight = (epoch / n_epochs) * physics_weight
            loss = data_loss + epoch_physics_weight * pde_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_data_loss += data_loss.item()
            running_pde_loss += pde_loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"    Epoch is {(n_batches / batches_per_epoch) * 100:.2f}% complete")

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

        if no_improvement_count >= 15:
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
            n_val_batches = 0

            for batch in val_loader:
                if n_val_batches > batches_per_epoch:
                    break

                x_val, y_val, z_val, ecg_val, T_val, sim_ids_val = [b.to(device) for b in batch]

                with torch.no_grad():
                    T_val_pred, c_val_pred = wrapper(x_val, y_val, z_val, ecg_val, sim_ids=sim_ids_val)
                    data_loss_val = nn.functional.mse_loss(T_val_pred, T_val)

                if sample_phys_each_epoch:
                    x_val_phys = (x_max - x_min) * torch.rand(phys_batch_size, 1, device=device) + x_min
                    y_val_phys = (y_max - y_min) * torch.rand(phys_batch_size, 1, device=device) + y_min
                    z_val_phys = (z_max - z_min) * torch.rand(phys_batch_size, 1, device=device) + z_min

                    x_val_phys.requires_grad_(True)
                    y_val_phys.requires_grad_(True)
                    z_val_phys.requires_grad_(True)

                    rand_idx = torch.randint(0, sim_ids_val.shape[0], (1,))
                    rand_idx = rand_idx.repeat(phys_batch_size)
                    val_phys_sim_ids = sim_ids_val[rand_idx].to(device)
                    ecg_val_phys = ecg_val[rand_idx]

                    residual_val = eikonal_residual(x_val_phys, y_val_phys, z_val_phys,
                                                    ecg_val_phys, wrapper, params, sim_ids=val_phys_sim_ids)
                    pde_loss_val = (residual_val**2).mean()
                else:
                    pde_loss_val = torch.tensor(0.0, device=device)

                total_val_loss = data_loss_val  # + physics_weight * pde_loss_val
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

            print_string += f" | Val Loss: {avg_val_loss:.4e}"

        print(print_string)

    return train_loss_hist, val_loss_hist, data_loss_hist, pde_loss_hist


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
