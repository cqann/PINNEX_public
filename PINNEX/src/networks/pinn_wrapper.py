import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedPINNWrapper(nn.Module):
    """
    Wraps the PINNWithECG model to cache ECG encodings for each sim_id
    during the current epoch.
    """

    def __init__(self, pinn_model, cache=True):
        super().__init__()
        self.model = pinn_model  # instance of PINNWithECG
        self._ecg_cache = {}     # sim_id -> cached z_ecg
        self.cache = cache

    def start_new_epoch(self):
        """
        Clear the ECG cache at the start of each epoch.
        """
        self._ecg_cache.clear()

    def forward(self, x, y, z, ecg, sim_ids=None):
        # 1) Compute PDE latent and normalize.
        z_pde = self.model.pde_encoder(x, y, z)

        # 2) Obtain ECG latent; use caching if sim_ids are provided.
        if sim_ids is None or not self.cache:
            z_ecg = self.model.ecg_encoder(ecg)
        else:
            batch_size = ecg.shape[0]
            latent_dim = self.model.ecg_encoder.latent_dim
            z_ecg = torch.zeros(batch_size, latent_dim, device=ecg.device)

            unique_sids = torch.unique(sim_ids)
            for sid in unique_sids:
                sid_int = int(sid.item())
                if sid_int not in self._ecg_cache:  # or True:
                    mask = (sim_ids == sid)
                    # Encode one representative ECG for the sim_id.
                    ecg_encoded = self.model.ecg_encoder(ecg[mask][:1])
                    self._ecg_cache[sid_int] = ecg_encoded
                z_ecg[sim_ids == sid] = self._ecg_cache[sid_int]

        # 3) Fuse latents using the model's fuse() method if available.
        if hasattr(self.model, "fuse"):
            z_fused = self.model.fuse(z_pde, z_ecg)
        else:
            z_fused = z_pde * z_ecg

        # 4) Decode and output activation time T and conduction velocity c.
        out = self.model.decoder(z_fused)
        T = F.softplus(out[:, :1])  # enforce T to be positive
        out_cv = out[:, 1:2]
        c = F.softplus(out_cv)
        return T, c
