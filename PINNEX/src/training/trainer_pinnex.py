import torch
from torch import nn
import torch.optim as optim

# Use the new eikonal residual function for the physics-based loss
import time
import numpy as np

from torch.utils.data import DataLoader  # Make sure DataLoader is imported
import math


def _initialize_training_state(model, lr, weight_decay, device):
    model = model.to(device)

    train_pde_encoder = True
    train_ecg_encoder = True
    train_decoder = True
    # Define components and their respective training flags
    # The 'decoder' flag controls decoder_body, decoder_out, and gate
    component_configs = []

    if hasattr(model, 'pde_encoder') and model.pde_encoder:
        component_configs.append((model.pde_encoder.parameters(), train_pde_encoder))

    if hasattr(model, 'ecg_encoder') and model.ecg_encoder:
        component_configs.append((model.ecg_encoder.parameters(), train_ecg_encoder))

    # Group all decoder-related parameters
    decoder_params_list = []
    for attr_name in ['decoder_body', 'decoder_out']:
        if hasattr(model, attr_name) and getattr(model, attr_name):
            decoder_params_list.extend(list(getattr(model, attr_name).parameters()))

    # Add gate parameters to decoder group if applicable
    if hasattr(model, 'gate') and model.gate and \
       hasattr(model, 'fusion_mode') and model.fusion_mode == "gated":
        decoder_params_list.extend(list(model.gate.parameters()))

    if decoder_params_list:  # Only add if there are any decoder params
        component_configs.append((iter(decoder_params_list), train_decoder))

    # Apply requires_grad settings
    for param_iterator, should_train in component_configs:
        for param in param_iterator:
            param.requires_grad = should_train

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    histories = {
        'train_loss': [], 'val_loss': [], 'data_loss': [],
        'pde_loss': [], 'cv_reg_loss': [], 'ne_loss': []  # Added Normal Enforcement loss history
    }

    lr_scheduler_state = {'best_train_loss': float('inf'), 'no_improvement_count': 0}
    early_stopping_state = {'best_val_loss': float('inf'), 'patience_counter': 0, 'best_model_state': None}

    return model, optimizer, histories, lr_scheduler_state, early_stopping_state


def _setup_physics_sampling_domain(
    collocation_points_filepath,
    train_loader,
    device,
    batches_per_epoch
):
    heart_collocation_points_with_normals = None  # Will store (x,y,z,nx,ny,nz)
    spatial_bounds = None

    try:
        print(f"Loading collocation points and normals from: {collocation_points_filepath}")
        # Expecting 6 columns: x, y, z, nx, ny, nz
        loaded_data = np.loadtxt(collocation_points_filepath)
        if loaded_data.ndim == 1 and loaded_data.shape[0] == 6:  # Handle single point with normal
            loaded_data = loaded_data.reshape(1, 6)
        elif loaded_data.ndim != 2 or loaded_data.shape[1] != 6:
            raise ValueError(
                f"Collocation file {collocation_points_filepath} must have 6 columns (X,Y,Z, NX,NY,NZ). Got shape {loaded_data.shape}")

        heart_collocation_points_with_normals = torch.from_numpy(loaded_data).float().to(device)
        print(
            f"Loaded {heart_collocation_points_with_normals.shape[0]} collocation points with normals")
        if heart_collocation_points_with_normals.shape[0] == 0:
            print("Warning: Loaded collocation points file is empty.")
            heart_collocation_points_with_normals = None
    except Exception as e:
        print(f"Error loading collocation points from {collocation_points_filepath}: {e}")
        print("CRITICAL: Proceeding without pre-loaded collocation points/normals.")
        heart_collocation_points_with_normals = None

    return heart_collocation_points_with_normals, spatial_bounds


def _get_physics_points_and_context(
    heart_collocation_data,  # Now contains points and normals
    spatial_bounds, phys_batch_size, device,
    current_data_batch_size, ecg_data_from_batch, sim_ids_from_batch
):
    x_phys, y_phys, z_phys = None, None, None
    normals_phys = None  # For (nx, ny, nz)
    ecg_for_physics, sim_ids_for_physics = None, None
    actual_phys_bs = 0

    if heart_collocation_data is not None and heart_collocation_data.shape[0] > 0:
        num_total_coll = heart_collocation_data.shape[0]
        actual_phys_bs = min(phys_batch_size, num_total_coll)

        if actual_phys_bs > 0:
            rand_indices = torch.randint(0, num_total_coll, (actual_phys_bs,), device=device)
            selected_data = heart_collocation_data[rand_indices]  # Shape (actual_phys_bs, 6)
            x_phys = selected_data[:, 0:1]
            y_phys = selected_data[:, 1:2]
            z_phys = selected_data[:, 2:3]
            normals_phys = selected_data[:, 3:6]  # Shape (actual_phys_bs, 3)

    if x_phys is not None and actual_phys_bs > 0:
        x_phys.requires_grad_(True)
        y_phys.requires_grad_(True)
        z_phys.requires_grad_(True)

        if current_data_batch_size > 0:
            single_rand_idx_tensor = torch.randint(0, current_data_batch_size, (1,), device=device)
            repeated_indices = single_rand_idx_tensor.repeat(actual_phys_bs)
            indices = torch.randint(0, current_data_batch_size, (actual_phys_bs,), device=device)
            ecg_for_physics = ecg_data_from_batch[indices]
            sim_ids_for_physics = sim_ids_from_batch[indices]
        else:
            ecg_for_physics, sim_ids_for_physics = None, None
            actual_phys_bs = 0
            x_phys, y_phys, z_phys, normals_phys = None, None, None, None
    else:
        x_phys, y_phys, z_phys, normals_phys = None, None, None, None
        ecg_for_physics, sim_ids_for_physics = None, None
        actual_phys_bs = 0

    return x_phys, y_phys, z_phys, normals_phys, ecg_for_physics, sim_ids_for_physics, actual_phys_bs


def _calculate_physics_loss_components(
    x_phys_norm, y_phys_norm, z_phys_norm,  # Renamed to clarify they are normalized
    normals_phys,  # Assumed to be normals of the PHYSICAL geometry
    ecg_for_physics, sim_ids_for_physics,
    wrapper, params_general,  # params_general might hold L_char_um if not in wrapper.model
    physics_weight, cv_reg_weight, normal_enforcement_weight,
    device,
    create_graph_for_physics,  # True for training, False for validation
    epsilon=1e-8
):
    pde_loss_val = torch.tensor(0.0, device=device)
    cv_reg_loss_val = torch.tensor(0.0, device=device)
    ne_loss_val = torch.tensor(0.0, device=device)

    # --- Retrieve scaling factors ---
    t_scale = wrapper.model.t_scale
    v_max_physical = wrapper.model.v_max
    L_char_um = wrapper.model.l_scale

    # --- Step 1: Get normalized predictions from the model ---
    T_norm, c_norm = wrapper(x_phys_norm, y_phys_norm, z_phys_norm, ecg_for_physics, sim_ids=sim_ids_for_physics)

    # Determine if individual gradient calculations will be needed
    # These booleans help in setting retain_graph correctly for validation
    calc_pde_grad = physics_weight > 0 and T_norm is not None and c_norm is not None
    calc_cv_grad = cv_reg_weight > 0 and c_norm is not None
    calc_ne_grad = normal_enforcement_weight > 0 and normals_phys is not None and T_norm is not None

    # --- Step 2: PDE Loss (Eikonal Residual) ---
    if calc_pde_grad:
        T_phys = T_norm * t_scale
        c_phys = c_norm * v_max_physical

        rg_pde = create_graph_for_physics or (not create_graph_for_physics and (calc_cv_grad or calc_ne_grad))

        grad_T_phys_wrt_norm_outputs = torch.autograd.grad(
            outputs=T_phys, inputs=[x_phys_norm, y_phys_norm, z_phys_norm],
            grad_outputs=torch.ones_like(T_phys), create_graph=create_graph_for_physics,
            retain_graph=rg_pde, allow_unused=True
        )
        dTphys_dxn = grad_T_phys_wrt_norm_outputs[0] if grad_T_phys_wrt_norm_outputs[0] is not None else torch.zeros_like(
            x_phys_norm)
        dTphys_dyn = grad_T_phys_wrt_norm_outputs[1] if grad_T_phys_wrt_norm_outputs[1] is not None else torch.zeros_like(
            y_phys_norm)
        dTphys_dzn = grad_T_phys_wrt_norm_outputs[2] if grad_T_phys_wrt_norm_outputs[2] is not None else torch.zeros_like(
            z_phys_norm)

        # These are gradients of T_phys w.r.t. NORMALIZED coordinates.
        # To get gradients of T_phys w.r.t. PHYSICAL coordinates, divide by L_char_um.

        dTphys_dxphys = dTphys_dxn / L_char_um
        dTphys_dyphys = dTphys_dyn / L_char_um
        dTphys_dzphys = dTphys_dzn / L_char_um

        # Now calculate the magnitude of the PHYSICAL gradient of T_phys
        grad_T_phys_physical_mag_sq = dTphys_dxphys**2 + dTphys_dyphys**2 + dTphys_dzphys**2
        grad_T_phys_physical_mag = torch.sqrt(grad_T_phys_physical_mag_sq + epsilon)  # Units: time/length

        # Eikonal residual: |∇_phys T_phys| - 1/c_phys
        # Units: (time/length) - (time/length)
        residual = grad_T_phys_physical_mag - 1.0 / (c_phys + epsilon)
        pde_loss_val = (residual**2).mean()

    # --- Step 3: Conduction Velocity (CV) Regularization Loss (on normalized c_norm) ---
    if calc_cv_grad:
        # retain_graph logic:
        # If training, True.
        # If validation, True if NE grad will follow this one.
        rg_cv = create_graph_for_physics or (not create_graph_for_physics and calc_ne_grad)

        grad_cnorm_wrt_norm_outputs = torch.autograd.grad(  # This was the failing line
            outputs=c_norm, inputs=[x_phys_norm, y_phys_norm, z_phys_norm],
            grad_outputs=torch.ones_like(c_norm), create_graph=create_graph_for_physics,
            retain_graph=rg_cv, allow_unused=True
        )
        dcn_dxn = grad_cnorm_wrt_norm_outputs[0] if grad_cnorm_wrt_norm_outputs[0] is not None else torch.zeros_like(
            x_phys_norm)
        dcn_dyn = grad_cnorm_wrt_norm_outputs[1] if grad_cnorm_wrt_norm_outputs[1] is not None else torch.zeros_like(
            y_phys_norm)
        dcn_dzn = grad_cnorm_wrt_norm_outputs[2] if grad_cnorm_wrt_norm_outputs[2] is not None else torch.zeros_like(
            z_phys_norm)
        grad_cnorm_norm_sq = dcn_dxn**2 + dcn_dyn**2 + dcn_dzn**2
        cv_reg_loss_val = grad_cnorm_norm_sq.mean()

    # --- Step 4: Normal Enforcement (NE) Loss (using T_norm and physical normals) ---
    if calc_ne_grad:
        # retain_graph logic:
        # If training, True.
        # If validation, False (as it's the last possible grad calculation in this sequence).
        rg_ne = create_graph_for_physics  # Effectively False if create_graph_for_physics is False

        grad_Tnorm_wrt_norm_outputs = torch.autograd.grad(
            outputs=T_norm, inputs=[x_phys_norm, y_phys_norm, z_phys_norm],
            grad_outputs=torch.ones_like(T_norm), create_graph=create_graph_for_physics,
            retain_graph=rg_ne, allow_unused=True
        )
        dTnorm_dxn = grad_Tnorm_wrt_norm_outputs[0] if grad_Tnorm_wrt_norm_outputs[0] is not None else torch.zeros_like(
            x_phys_norm)
        dTnorm_dyn = grad_Tnorm_wrt_norm_outputs[1] if grad_Tnorm_wrt_norm_outputs[1] is not None else torch.zeros_like(
            y_phys_norm)
        dTnorm_dzn = grad_Tnorm_wrt_norm_outputs[2] if grad_Tnorm_wrt_norm_outputs[2] is not None else torch.zeros_like(
            z_phys_norm)

        if L_char_um == 0:  # Avoid division by zero if L_char_um is not properly set or is zero
            dot_product_physical_sense = torch.tensor(0.0, device=device)  # Or handle error
            print("Warning: L_char_um is zero in NE loss calculation.")
        else:
            dot_product_physical_sense = (
                (dTnorm_dxn / L_char_um) * normals_phys[:, 0:1] +
                (dTnorm_dyn / L_char_um) * normals_phys[:, 1:2] +
                (dTnorm_dzn / L_char_um) * normals_phys[:, 2:3]
            )
        ne_loss_val = (dot_product_physical_sense**2).mean()

    return pde_loss_val, cv_reg_loss_val, ne_loss_val


def _train_one_epoch(
    wrapper, model, train_loader, optimizer, params_general, device, epoch_num, tot_epochs,
    heart_colloc_data, spatial_bounds,  # heart_colloc_data has points+normals
    physics_weight, cv_reg_weight, normal_enforcement_weight,  # Added NE weight
    phys_batch_size, batches_per_epoch, sample_phys_each_epoch,
):
    model.train()
    wrapper.start_new_epoch()

    running_losses = {'total': 0.0, 'data': 0.0, 'pde': 0.0, 'cv_reg': 0.0, 'ne': 0.0}  # Added NE loss
    n_batches_processed = 0

    effective_loader_len = len(train_loader)
    if batches_per_epoch > 0:
        effective_loader_len = min(len(train_loader), batches_per_epoch)

    for batch_idx, batch_content in enumerate(train_loader):
        wrapper.start_new_epoch()

        if batches_per_epoch > 0 and n_batches_processed >= batches_per_epoch:
            break

        x_d, y_d, z_d, ecg_d, T_d, V_d, sim_ids_d = [b.to(device) for b in batch_content]
        current_data_batch_size = x_d.size(0)

        optimizer.zero_grad()

        T_pred_data, V_pred_data = wrapper(x_d, y_d, z_d, ecg_d, sim_ids=sim_ids_d)
        T_data_loss = nn.functional.mse_loss(T_pred_data, T_d)
        V_data_loss = nn.functional.mse_loss(V_pred_data, V_d)
        data_loss = 1 * T_data_loss + 0 * V_data_loss

        pde_loss_batch = torch.tensor(0.0, device=device)
        cv_reg_loss_batch = torch.tensor(0.0, device=device)
        ne_loss_batch = torch.tensor(0.0, device=device)  # Added NE loss batch

        if sample_phys_each_epoch and current_data_batch_size > 0:
            x_phys, y_phys, z_phys, normals_phys, ecg_phys_ctx, sim_ids_phys_ctx, actual_phys_bs = \
                _get_physics_points_and_context(
                    heart_colloc_data, spatial_bounds, phys_batch_size, device,
                    current_data_batch_size, ecg_d, sim_ids_d
                )
            if actual_phys_bs > 0:  # Check if points were actually sampled
                pde_loss_batch, cv_reg_loss_batch, ne_loss_batch = _calculate_physics_loss_components(
                    x_phys, y_phys, z_phys, normals_phys, ecg_phys_ctx, sim_ids_phys_ctx,
                    wrapper, params_general, physics_weight, cv_reg_weight, normal_enforcement_weight, device,
                    create_graph_for_physics=True
                )

        start = 1 - 0.5
        weight_tuning = - start + (2.71828)**(math.log(1 + start) * (epoch_num / tot_epochs))

        loss = data_loss + \
            physics_weight * pde_loss_batch * weight_tuning + \
            cv_reg_weight * cv_reg_loss_batch * weight_tuning + \
            normal_enforcement_weight * ne_loss_batch * weight_tuning  # Added NE loss to total
        loss.backward()
        optimizer.step()

        running_losses['total'] += loss.item()
        running_losses['data'] += data_loss.item()
        running_losses['pde'] += pde_loss_batch.item()
        running_losses['cv_reg'] += cv_reg_loss_batch.item()
        running_losses['ne'] += ne_loss_batch.item()  # Accumulate NE loss
        n_batches_processed += 1

        if batch_idx > 0 and batch_idx % 50 == 0:

            print(f"\rEpoch {epoch_num}, Batch {batch_idx}/{effective_loader_len} processed...", end="", flush=True)

    avg_losses = {k: v / n_batches_processed if n_batches_processed > 0 else 0 for k, v in running_losses.items()}
    return avg_losses


def _validate_one_epoch(
    wrapper, model, val_loader, params_general, device,
    heart_colloc_data, spatial_bounds,  # heart_colloc_data has points+normals
    physics_weight, cv_reg_weight, normal_enforcement_weight,  # Added NE weight
    phys_batch_size, batches_per_epoch, sample_phys_each_epoch
):
    model.eval()
    running_val_losses = {'total': 0.0, 'data': 0.0, 'pde': 0.0, 'cv_reg': 0.0, 'ne': 0.0, 're': 0.0}  # Added NE
    n_val_batches_processed = 0

    effective_loader_len = len(val_loader)
    if batches_per_epoch > 0:
        effective_loader_len = min(len(val_loader), batches_per_epoch)

    with torch.no_grad():
        for val_batch_idx, batch_val_content in enumerate(val_loader):
            if batches_per_epoch > 0 and n_val_batches_processed >= batches_per_epoch:
                break

            x_v, y_v, z_v, ecg_v, T_v, V_v, sim_ids_v = [b.to(device) for b in batch_val_content]
            current_val_batch_size = x_v.size(0)

            T_val_pred, V_val_pred = wrapper(x_v, y_v, z_v, ecg_v, sim_ids=sim_ids_v)
            T_val_data_loss = nn.functional.mse_loss(T_val_pred, T_v)
            V_val_data_loss = nn.functional.mse_loss(V_val_pred, V_v)
            val_data_loss = 1 * T_val_data_loss + 0 * V_val_data_loss

            relative_error = torch.norm(T_val_pred.squeeze() - T_v.squeeze()) / (torch.norm(T_v.squeeze()) + 1e-8)
            running_val_losses['re'] += relative_error.item()

            val_pde_loss_b = torch.tensor(0.0, device=device)
            val_cv_reg_loss_b = torch.tensor(0.0, device=device)
            val_ne_loss_b = torch.tensor(0.0, device=device)  # Added NE

            if sample_phys_each_epoch and current_val_batch_size > 0:
                x_val_phys, y_val_phys, z_val_phys, normals_val_phys, ecg_val_phys_ctx, sim_ids_val_phys_ctx, actual_val_phys_bs = \
                    _get_physics_points_and_context(
                        heart_colloc_data, spatial_bounds, phys_batch_size, device,
                        current_val_batch_size, ecg_v, sim_ids_v
                    )
                if actual_val_phys_bs > 0:  # Check if points were actually sampled
                    with torch.enable_grad():
                        val_pde_calc, val_cv_reg_calc, val_ne_calc = _calculate_physics_loss_components(
                            x_val_phys, y_val_phys, z_val_phys, normals_val_phys,
                            ecg_val_phys_ctx, sim_ids_val_phys_ctx,
                            wrapper, params_general, physics_weight, cv_reg_weight, normal_enforcement_weight, device,
                            create_graph_for_physics=False  # create_graph=False for validation
                        )
                    val_pde_loss_b = val_pde_calc.detach()
                    val_cv_reg_loss_b = val_cv_reg_calc.detach()
                    val_ne_loss_b = val_ne_calc.detach()  # Detach NE loss

            current_val_total_loss = val_data_loss + \
                physics_weight * val_pde_loss_b + \
                cv_reg_weight * val_cv_reg_loss_b + \
                normal_enforcement_weight * val_ne_loss_b  # Add NE loss
            running_val_losses['total'] += current_val_total_loss.item()
            running_val_losses['data'] += val_data_loss.item()
            running_val_losses['pde'] += val_pde_loss_b.item()
            running_val_losses['cv_reg'] += val_cv_reg_loss_b.item()
            running_val_losses['ne'] += val_ne_loss_b.item()  # Accumulate NE loss
            n_val_batches_processed += 1

    avg_val_losses = {k: v / n_val_batches_processed if n_val_batches_processed >
                      0 else 0 for k, v in running_val_losses.items()}
    return avg_val_losses


def _update_lr_scheduler_state(optimizer, avg_epoch_loss, lr_scheduler_state, min_lr):
    lr_patience = 6
    lr_factor = 5.0
    best_train_loss = lr_scheduler_state['best_train_loss']
    no_improvement_count = lr_scheduler_state['no_improvement_count']
    lr_reduced_to_min = False

    if avg_epoch_loss < best_train_loss:
        best_train_loss = avg_epoch_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= lr_patience:
        new_lr_val = -1  # Placeholder
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] / lr_factor, min_lr)
            new_lr_val = param_group['lr']  # Get the actual new LR
        if new_lr_val != -1:  # Log if any change happened or attempt was made
            print(f"Training loss stagnated. Reducing learning rate to {new_lr_val:.4e}")
        no_improvement_count = 0
        if abs(optimizer.param_groups[0]['lr'] - min_lr) < 1e-9:
            print("Learning rate reached minimum.")
            lr_reduced_to_min = True

    lr_scheduler_state['best_train_loss'] = best_train_loss
    lr_scheduler_state['no_improvement_count'] = no_improvement_count
    return lr_scheduler_state, lr_reduced_to_min


def _check_early_stopping_condition(avg_val_loss, early_stopping_state, model, patience_val):
    best_val_loss = early_stopping_state['best_val_loss']
    patience_counter = early_stopping_state['patience_counter']
    best_model_state = early_stopping_state['best_model_state']
    stop_early = False

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
        print(f"*** New best validation loss: {best_val_loss:.4e} ***")
    else:
        patience_counter += 1
        if patience_counter >= patience_val:
            print(f"Validation loss did not improve for {patience_val} validation steps. Stopping early.")
            stop_early = True

    early_stopping_state['best_val_loss'] = best_val_loss
    early_stopping_state['patience_counter'] = patience_counter
    early_stopping_state['best_model_state'] = best_model_state
    return early_stopping_state, stop_early

# --- Main Training Function (Matches Original Signature + new weight) ---


def train_pinnex_minibatch_with_ecg(wrapper,
                                    train_loader,
                                    val_loader,
                                    params,
                                    n_epochs=10000,
                                    lr=1e-4,
                                    weight_decay=1e-4,
                                    physics_weight=1e-4,
                                    cv_reg_weight=1e-8,
                                    normal_enforcement_weight=0.0,  # New parameter
                                    batches_per_epoch=1000,
                                    phys_batch_size=1000,
                                    val_every_n_epochs=5,
                                    device='cpu',
                                    collocation_points_filepath="coll_points.dat"):

    model_instance, optimizer, histories, lr_scheduler_state, early_stopping_state = \
        _initialize_training_state(wrapper.model, lr, weight_decay, device)

    sample_phys_each_epoch = (phys_batch_size > 0) and \
                             (physics_weight > 0 or cv_reg_weight > 0 or (
                                 normal_enforcement_weight > 0))

    heart_colloc_data, spatial_bds = None, None  # heart_colloc_data now holds points and normals
    if sample_phys_each_epoch:
        heart_colloc_data, spatial_bds = _setup_physics_sampling_domain(
            collocation_points_filepath, train_loader, device, batches_per_epoch
        )

    min_lr_val = 1e-7
    early_stopping_patience_val = 16

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        avg_train_losses = _train_one_epoch(
            wrapper, model_instance, train_loader, optimizer, params, device, epoch + 1, n_epochs,
            heart_colloc_data, spatial_bds,
            physics_weight, cv_reg_weight, normal_enforcement_weight,  # Pass NE weight
            phys_batch_size, batches_per_epoch, sample_phys_each_epoch,
        )

        histories['train_loss'].append(avg_train_losses['total'])
        histories['data_loss'].append(avg_train_losses['data'])
        histories['pde_loss'].append(avg_train_losses['pde'])
        histories['cv_reg_loss'].append(avg_train_losses['cv_reg'])
        histories['ne_loss'].append(avg_train_losses['ne'])  # Store NE loss history

        print_string = (f"\n[Epoch {epoch + 1}/{n_epochs}] Train Loss: {avg_train_losses['total']:.4e} | "
                        f"Data: {avg_train_losses['data']:.4e} | PDE: {avg_train_losses['pde']:.4e} | "
                        f"CV-Reg: {avg_train_losses['cv_reg']:.4e} | NE: {avg_train_losses['ne']:.4e}")

        lr_scheduler_state, lr_at_min = _update_lr_scheduler_state(
            optimizer, avg_train_losses['total'], lr_scheduler_state, min_lr_val
        )

        if (epoch + 1) % val_every_n_epochs == 0:
            avg_val_losses = _validate_one_epoch(
                wrapper, model_instance, val_loader, params, device,
                heart_colloc_data, spatial_bds,
                physics_weight, cv_reg_weight, normal_enforcement_weight,  # Pass NE weight
                phys_batch_size, batches_per_epoch, sample_phys_each_epoch)
            histories['val_loss'].append(avg_val_losses['total'])

            print_string += (f"\n| Val Loss: {avg_val_losses['total']:.4e} (Data: {avg_val_losses['data']:.4e}, "
                             f"PDE: {avg_val_losses['pde']:.4e}, CV-Reg: {avg_val_losses['cv_reg']:.4e}, NE: {avg_val_losses['ne']:.4e}) "
                             f"| RE: {avg_val_losses['re']:.4e}")

            early_stopping_state, stop_early_flag = _check_early_stopping_condition(
                avg_val_losses['total'], early_stopping_state, model_instance, early_stopping_patience_val
            )
            if stop_early_flag:
                break

        epoch_time = time.time() - epoch_start_time
        print_string += f" | Time: {epoch_time:.2f}s"
        print_string += "\n------------------------------------------------------------------------------------------------------------------------------------------"

        print(print_string)

        if lr_at_min:
            current_lr = optimizer.param_groups[0]['lr']
            if abs(current_lr - min_lr_val) < 1e-9:
                print("Learning rate at minimum. Stopping training early.")
                break

    if early_stopping_state['best_model_state'] is not None:
        print(
            f"Training finished. Loading model with best validation loss: {early_stopping_state['best_val_loss']:.4e}")
        model_instance.load_state_dict(early_stopping_state['best_model_state'])
    else:
        print("Training finished. No best model state recorded. Using last model state.")

    return histories['train_loss'], histories['val_loss'], histories['data_loss'], histories['pde_loss']
