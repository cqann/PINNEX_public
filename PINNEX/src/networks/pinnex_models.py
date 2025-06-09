import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# -----------------------------
# PDE_Encoder with:
#   - Optional Fourier features
#   - Residual (skip) connections
#   - Optional layer normalization
# -----------------------------


class PDE_Encoder(nn.Module):
    """
    MLP to encode (x, y, z) -> z_PDE for the eikonal equation,
    with optional Fourier features, skip connections, and normalization.
    """

    def __init__(
        self,
        in_dim=3,  # changed from 4 to 3 (time is not used)
        latent_dim=64,
        hidden_architecture=[16, 16, 16],
        use_fourier_features=False,
        fourier_embedding_size=8,
        use_skip_connections=True,
        use_layernorm=False
    ):
        super().__init__()

        # ADDED: Optional Fourier features
        # For each coordinate, we'll generate sin/cos expansions
        # so effectively your input dimension can jump from `in_dim` to `in_dim*(2*fourier_embedding_size + 1)`.
        self.use_fourier_features = use_fourier_features
        self.fourier_embedding_size = fourier_embedding_size
        if use_fourier_features:
            self.basis_freqs = nn.Parameter(
                2.0 * math.pi * torch.rand(in_dim, fourier_embedding_size),
                requires_grad=False
            )

        # Figure out the effective input dimension after Fourier features
        effective_in_dim = in_dim if not use_fourier_features else (in_dim * (2 * fourier_embedding_size + 1))

        # Build an MLP with optional skip connections between each pair of layers
        layers = []
        prev_size = effective_in_dim
        self.skip_connections = use_skip_connections

        for out_size in hidden_architecture:
            layers.append(ResidualBlock(prev_size, out_size, skip=use_skip_connections))  # ADDED
            prev_size = out_size

        self.network = nn.Sequential(*layers)

        # ADDED: optional layernorm after the final hidden layer
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = nn.LayerNorm(prev_size)

        self.output_layer = nn.Linear(prev_size, latent_dim)

    def _fourier_embed(self, coords):
        """
        Expand each input coordinate x -> [x, sin(b1*x), cos(b1*x), sin(b2*x), cos(b2*x), ...]
        """
        # coords shape: (N, in_dim)
        embedded = [coords]  # always keep the raw coords
        for i in range(self.fourier_embedding_size):
            freq = self.basis_freqs[:, i]  # shape (in_dim,)
            # broadcast multiply
            proj = coords * freq  # (N, in_dim)
            embedded.append(torch.sin(proj))
            embedded.append(torch.cos(proj))
        return torch.cat(embedded, dim=1)  # shape (N, in_dim*(2*fourier_embed + 1))

    def forward(self, x, y, z):
        # Modified: remove time coordinate from the input
        inp = torch.cat((x, y, z), dim=1)
        if self.use_fourier_features:
            inp = self._fourier_embed(inp)
        out = self.network(inp)
        if self.use_layernorm:
            out = self.norm(out)
        z_pde = self.output_layer(out)
        return z_pde


class ECG_Encoder(nn.Module):
    """
    1D CNN-based encoder for ECG signals.
    Uses hidden_architecture for the fully connected layers after the CNN part.
    Now has a parameter `n_blocks` to repeat the CNN block structure.
    """

    def __init__(self, n_leads=12, seq_len=1000, latent_dim=64,
                 hidden_architecture=[128], n_blocks=2):
        super().__init__()

        # Create repeated blocks
        # Each block: BN -> ReLU -> Conv -> BN -> ReLU -> Dropout -> Conv -> MaxPool
        self.blocks = nn.ModuleList()
        current_channels = n_leads
        # Define desired channels for each block
        desired_channels = [16, 16, 32, 32]  # For n_blocks=2; adjust accordingly if n_blocks changes
        for i in range(n_blocks):
            out_channels = desired_channels[i]
            block = nn.Sequential(
                nn.BatchNorm1d(current_channels),
                nn.ReLU(),
                nn.Conv1d(current_channels, out_channels, kernel_size=4, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.blocks.append(block)
            current_channels = out_channels

        self.latent_dim = latent_dim

        # Determine output dimension after all blocks, via a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, seq_len)
            x = dummy
            for block in self.blocks:
                x = block(x)
            flattened_dim = x.view(1, -1).size(1)

        # Build the fully connected network
        fc_layers = []
        if hidden_architecture:
            fc_layers.append(nn.Linear(flattened_dim, hidden_architecture[0]))
            fc_layers.append(nn.ReLU())
            for i in range(len(hidden_architecture) - 1):
                fc_layers.append(nn.Linear(hidden_architecture[i], hidden_architecture[i + 1]))
                fc_layers.append(nn.ReLU())
            last_dim = hidden_architecture[-1]
        else:
            last_dim = flattened_dim

        fc_layers.append(nn.Linear(last_dim, latent_dim))
        self.fc_net = nn.Sequential(*fc_layers)

    def forward(self, ecg):
        x = ecg
        for block in self.blocks:
            x = block(x)

        x = x.view(x.size(0), -1)
        z_ecg = self.fc_net(x)
        return z_ecg

# -----------------------------
# A small ResidualBlock for skip connections
# -----------------------------


class ECG_Encoder_With_Skips(nn.Module):
    """
    1D CNN-based encoder for ECG signals with residual skip connections.
    Uses hidden_architecture for the fully connected layers after the CNN part.
    Has a parameter `n_blocks` to repeat the CNN block structure.
    Each block in the loop will have a residual skip connection.
    """

    def __init__(self, n_leads=12, seq_len=1000, latent_dim=64,
                 hidden_architecture=[128], n_blocks=2,
                 desired_channels_per_block=None):  # e.g., [16, 32] for n_blocks=2
        super().__init__()
        self.n_blocks = n_blocks

        if desired_channels_per_block is None:
            # Provide a default or raise an error if not specified.
            # This example default creates a simple channel progression.
            # It's best if the user specifies this based on their needs.
            if n_blocks == 1:
                desired_channels_per_block = [16]
            elif n_blocks == 2:
                desired_channels_per_block = [16, 32]
            elif n_blocks == 3:
                desired_channels_per_block = [16, 32, 64]
            elif n_blocks == 4:  # Example similar to original code's comment context

                desired_channels_per_block = [16, 16, 32, 32]
                desired_channels_per_block = [16, 32, 64, 128]

            else:  # Generic fallback for other n_blocks values
                ch = 16
                desired_channels_per_block = []
                for i in range(n_blocks):
                    desired_channels_per_block.append(ch)
                    if i % 2 == 1 and ch < 128:
                        ch *= 2  # Increase channels every two blocks
            print(f"Using default desired_channels_per_block: {desired_channels_per_block} for n_blocks={n_blocks}")

        if len(desired_channels_per_block) != n_blocks:
            raise ValueError(f"Length of desired_channels_per_block ({len(desired_channels_per_block)}) "
                             f"must be equal to n_blocks ({n_blocks})")

        self.blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.skip_relus = nn.ModuleList()  # To add ReLU after skip addition

        current_channels_input_to_block = n_leads
        for i in range(n_blocks):
            out_channels_from_block = desired_channels_per_block[i]

            # Main block definition
            # IMPORTANT: Using padding='same' in Conv1d to preserve sequence length for skip connections.
            # If using kernel_size=4, stride=1, padding=1 (as in original), sequence length
            # changes by -1 per conv, making direct addition with skip connection problematic.
            # 'padding="same"' requires PyTorch 1.9+.
            # If not available, adjust kernel/padding (e.g., kernel_size=3, padding=1)
            # or implement more complex skip projection/cropping.
            block = nn.Sequential(
                nn.BatchNorm1d(current_channels_input_to_block),
                nn.ReLU(),
                nn.Conv1d(current_channels_input_to_block, out_channels_from_block,
                          kernel_size=20, stride=1, padding='same', bias=False),  # Changed padding
                nn.BatchNorm1d(out_channels_from_block),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Conv1d(out_channels_from_block, out_channels_from_block,
                          kernel_size=20, stride=1, padding='same', bias=False),  # Changed padding
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.blocks.append(block)

            # Skip connection projection for the input to this block (identity)
            # Transforms 'identity' to match the shape of 'block(identity)'
            projection_ops = []
            # 1. Channel matching: If input channels to block != output channels from block
            if current_channels_input_to_block != out_channels_from_block:
                projection_ops.extend([
                    nn.Conv1d(current_channels_input_to_block, out_channels_from_block,
                              kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm1d(out_channels_from_block)
                ])

            # 2. Downsampling: Always match the MaxPool1d in the main block path
            projection_ops.append(nn.MaxPool1d(kernel_size=2, stride=2))

            self.skip_projections.append(nn.Sequential(*projection_ops))
            self.skip_relus.append(nn.ReLU())  # ReLU after each skip addition

            current_channels_input_to_block = out_channels_from_block  # Update for the next block's input

        self.latent_dim = latent_dim

        # Determine output dimension after all blocks and skips, via a dummy pass
        with torch.no_grad():
            dummy_ecg = torch.zeros(1, n_leads, seq_len)
            x_dummy = dummy_ecg
            for i in range(n_blocks):
                identity_dummy = x_dummy
                x_dummy_main = self.blocks[i](identity_dummy)
                projected_identity_dummy = self.skip_projections[i](identity_dummy)

                # Ensure consistent sequence lengths for addition in dummy pass
                # This might be needed if padding='same' isn't perfect or if seq_len is odd
                if x_dummy_main.size(2) != projected_identity_dummy.size(2):
                    # Simple cropping strategy: crop the larger one from the end
                    min_len = min(x_dummy_main.size(2), projected_identity_dummy.size(2))
                    x_dummy_main = x_dummy_main[:, :, :min_len]
                    projected_identity_dummy = projected_identity_dummy[:, :, :min_len]

                x_dummy = x_dummy_main + projected_identity_dummy
                x_dummy = self.skip_relus[i](x_dummy)
            flattened_dim = x_dummy.view(1, -1).size(1)

        # Build the fully connected network
        fc_layers = []
        current_fc_dim = flattened_dim
        if hidden_architecture:
            fc_layers.append(nn.Linear(current_fc_dim, hidden_architecture[0]))
            fc_layers.append(nn.ReLU())
            current_fc_dim = hidden_architecture[0]
            for i in range(len(hidden_architecture) - 1):
                fc_layers.append(nn.Linear(current_fc_dim, hidden_architecture[i + 1]))
                fc_layers.append(nn.ReLU())
                current_fc_dim = hidden_architecture[i + 1]

        fc_layers.append(nn.Linear(current_fc_dim, latent_dim))
        self.fc_net = nn.Sequential(*fc_layers)

    def forward(self, ecg):
        x = ecg
        for i in range(self.n_blocks):
            identity = x
            x_main = self.blocks[i](identity)  # Output of the main block path

            # Project identity to match shape of x_main for the skip connection
            identity_projected = self.skip_projections[i](identity)

            # Ensure sequence lengths match before addition (critical)
            # padding='same' should handle this for even sequence lengths.
            # If sequence lengths can be odd or there are slight discrepancies:
            if x_main.size(2) != identity_projected.size(2):
                # This can happen if input seq_len to MaxPool is odd.
                # e.g. floor(L/2) vs floor(L/2)
                # However, if convs output L' and then maxpool, it's floor(L'/2)
                # If skip is on L, it's floor(L/2). If L' != L, they might differ.
                # With padding='same', L' should be L.
                # A robust way is to crop to the minimum length if a mismatch occurs.
                min_len = min(x_main.size(2), identity_projected.size(2))
                x_main = x_main[:, :, :min_len]
                identity_projected = identity_projected[:, :, :min_len]

            x = x_main + identity_projected  # Add skip connection
            x = self.skip_relus[i](x)      # Apply ReLU after addition

        x = x.view(x.size(0), -1)  # Flatten
        z_ecg = self.fc_net(x)
        return z_ecg


class ResidualBlock(nn.Module):  # ADDED
    """
    A single MLP block with skip connection if `skip=True`.
    """

    def __init__(self, in_size, out_size, skip=True):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.act = nn.SiLU()
        # self.dropout = nn.Dropout(p=0.25)
        self.skip = (skip and in_size == out_size)  # can only skip if shapes match

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        # out = self.dropout(out)
        if self.skip:
            out = out + x
        return out

# -----------------------------
# PINNWithECG:
# Add a small 'fusion' module to combine z_pde and z_ecg
# via concat or gating, plus an option for skip connections in decoder
# -----------------------------

# -----------------------------
# Modified PINNWithECG model
# -----------------------------


class PINNWithECG(nn.Module):
    """
    Full model for the eikonal equation with:
      - PDE encoder: z_pde = PDE_Encoder(x, y, z)
      - ECG encoder: z_ecg = ECG_Encoder(ecg)
      - A fusion step to combine z_pde and z_ecg
      - A decoder that outputs activation time T and conduction velocity c
    """

    def __init__(
        self,
        params,
        pde_latent_dim=64,
        ecg_latent_dim=64,
        pde_hidden_architecture=[16, 16, 16],
        ecg_hidden_architecture=[128],
        decoder_hidden_architecture=[64, 64],
        n_leads=12,
        n_blocks=4,
        seq_len=1000,
        fusion_mode="gated",
        decoder_use_skip=False
    ):
        super().__init__()
        self.t_scale = params["t_scale"]
        self.v_max = params["v_max"]
        self.l_scale = params["l_scale"]

        self.pde_encoder = PDE_Encoder(
            in_dim=3,  # now only spatial coordinates (x, y, z)
            latent_dim=pde_latent_dim,
            hidden_architecture=pde_hidden_architecture,
            use_fourier_features=True,
            fourier_embedding_size=8,
            use_skip_connections=True,
            use_layernorm=True
        )
        self.ecg_encoder = ECG_Encoder_With_Skips(
            n_leads=n_leads,
            seq_len=seq_len,
            latent_dim=ecg_latent_dim,
            hidden_architecture=ecg_hidden_architecture,
            n_blocks=n_blocks
        )

        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            decoder_input_dim = pde_latent_dim + ecg_latent_dim
        elif fusion_mode == "mul":
            decoder_input_dim = pde_latent_dim
        elif fusion_mode == "gated":
            self.gate = nn.Linear(ecg_latent_dim, pde_latent_dim)
            decoder_input_dim = pde_latent_dim
        else:
            raise ValueError("Invalid fusion_mode. Choose from ['mul','concat','gated']")

        # Build decoder with skip connections
        decoder_layers = []
        prev_size = decoder_input_dim
        for out_size in decoder_hidden_architecture:
            decoder_layers.append(ResidualBlock(prev_size, out_size, skip=decoder_use_skip))
            prev_size = out_size
        self.decoder_body = nn.Sequential(*decoder_layers)
        self.decoder_out = nn.Linear(prev_size, 2)  # outputs (T, c)

    def fuse(self, z_pde, z_ecg):
        if self.fusion_mode == "concat":
            return torch.cat([z_pde, z_ecg], dim=1)
        elif self.fusion_mode == "mul":
            return z_pde * z_ecg
        elif self.fusion_mode == "gated":
            gate_vals = torch.sigmoid(self.gate(z_ecg))
            return z_pde * gate_vals

    @property
    def decoder(self):
        return lambda z: self.decoder_out(self.decoder_body(z))

    def forward(self, x, y, z, ecg):
        # Modified: no time input; only spatial coordinates are used.
        z_pde = self.pde_encoder(x, y, z)
        z_ecg = self.ecg_encoder(ecg)

        z_fused = self.fuse(z_pde, z_ecg)
        out = self.decoder(z_fused)
        # In the forward method of PINNWithECG, modify the output activation as follows:
        T = F.softplus(out[:, :1])  # enforce T to be positive
        out_cv = out[:, 1:2]
        c = F.softplus(out_cv)
        return T, c
