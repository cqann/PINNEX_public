import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


class SpatiotemporalECGDataset(Dataset):
    """
    A PyTorch Dataset that keeps:
      - a large spatial table with columns: [x, y, z, T, sim_id]
      - a dictionary of (sim_id -> ecg_tensor)
    so that we don't replicate the ECG data for every row.
    """

    def __init__(self, df_spatial, ecg_dict):
        """
        Parameters
        ----------
        df_spatial : pd.DataFrame
            Contains columns [x, y, z, T, sim_id]
        ecg_dict : Dict[int, torch.Tensor]
            Maps sim_id -> a single ECG tensor of shape (n_leads, seq_len).
        """
        self.x = torch.tensor(df_spatial['x'].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(df_spatial['y'].values, dtype=torch.float32).view(-1, 1)
        self.z = torch.tensor(df_spatial['z'].values, dtype=torch.float32).view(-1, 1)
        self.T = torch.tensor(df_spatial['T'].values, dtype=torch.float32).view(-1, 1)

        if 'cv' in df_spatial.columns:
            self.V = torch.tensor(df_spatial['cv'].values, dtype=torch.float32).view(-1, 1)
        else:
            # Create a tensor of NaNs with the same length as other features
            # and dtype float32.
            num_rows = len(df_spatial)
            self.V = torch.full((num_rows, 1), float(0), dtype=torch.float32)
            print("Warning: 'cv' column not found in df_spatial. Using 0 as a placeholder for V.")

        self.sim_id = torch.tensor(df_spatial['sim_id'].values, dtype=torch.int32).view(-1)

        # Save the dictionary for lookups
        self.ecg_dict = ecg_dict

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Grab spatial features and target activation time
        x_val = self.x[idx]
        y_val = self.y[idx]
        z_val = self.z[idx]
        T_val = self.T[idx]
        V_val = self.V[idx]
        sim_key = self.sim_id[idx].item()  # e.g. 1, 2, etc.

        # Look up the single ECG (n_leads, seq_len)
        ecg_tensor = self.ecg_dict[sim_key]

        return (x_val, y_val, z_val, ecg_tensor, T_val, V_val, sim_key)


def collate_fn(batch):
    x, y, z, ecg, T, V, sim_ids = zip(*batch)

    x = torch.stack(x)
    y = torch.stack(y)
    z = torch.stack(z)
    ecg = torch.stack(ecg)
    T = torch.stack(T)
    V = torch.stack(V)
    sim_ids = torch.tensor(sim_ids, dtype=torch.int32)

    return x, y, z, ecg, T, V, sim_ids


def load_spatiotemp_df(spatiotemp_path):
    """
    Reads the spatial parquet file containing columns: [x, y, z, T, sim_id].
    """
    df = pd.read_parquet(spatiotemp_path)
    df = df.reset_index(drop=True)
    # df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)

    return df


def load_ecg_dict(ecg_path):
    """
    Reads the ECG parquet file and returns a dictionary:
    sim_id -> torch.Tensor of shape (n_leads, seq_len).
    Handles nested list/array inconsistencies robustly.
    """
    df_ecg = pd.read_parquet(ecg_path).reset_index(drop=True)
    ecg_dict = {}

    for row in df_ecg.itertuples(index=False):
        sim_id = getattr(row, 'sim_id')
        ecg_obj = getattr(row, 'ecg')

        try:
            # Handle different possible nested formats
            if isinstance(ecg_obj, str):
                ecg_obj = eval(ecg_obj)  # convert string to list (if stored as string)
            if isinstance(ecg_obj, np.ndarray) and ecg_obj.dtype == object:
                ecg_array = np.array([np.array(lead, dtype=np.float32) for lead in ecg_obj])
            else:
                ecg_array = np.array(ecg_obj, dtype=np.float32)

            if ecg_array.ndim != 2:
                raise ValueError(f"ECG for sim_id {sim_id} is not 2D. Got shape: {ecg_array.shape}")

            ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32)
            ecg_dict[sim_id] = ecg_tensor

        except Exception as e:
            print(f"[ERROR] Could not parse ECG for sim_id {sim_id}: {e}")
            continue

    return ecg_dict


def split_train_val_df(df_spatial, val_ratio=0.2, random_sample=False, seed=None, group_by_delay_variant=False):
    """
    Splits the DataFrame into training and validation sets.

    Parameters:
    ----------
    df_spatial : pd.DataFrame
        DataFrame containing at least 'sim_id'.
    val_ratio : float
        Proportion of the dataset to allocate to the validation set.
    random_sample : bool
        If True, perform a standard random row-wise split, ignoring sim_id grouping.
        If False (default), split by sim_id or base_sim_id (if group_by_delay_variant is True).
    seed : int, optional
        Random seed for reproducible splits.
    group_by_delay_variant : bool
        If True, assumes the last digit of 'sim_id' is a delay variant.
        Splits will be made based on 'sim_id' excluding the last digit,
        ensuring all delay variants of a base simulation stay in the same set.
    """
    if random_sample:
        # Use the seed for df.sample's random_state for reproducible row-wise shuffling
        df_shuffled = df_spatial.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        num_val = int(len(df_shuffled) * val_ratio)
        df_val = df_shuffled.iloc[:num_val].copy()
        df_train = df_shuffled.iloc[num_val:].copy()
        print(f"Random sample split: {len(df_train)} train rows, {len(df_val)} val rows.")
    elif group_by_delay_variant:
        # Create a 'base_sim_id' by removing the last digit
        # Ensure sim_id is treated as integer for the division
        df_spatial['_base_sim_id_temp'] = (df_spatial['sim_id'].astype(int) // 10)
        unique_base_sim_ids = df_spatial['_base_sim_id_temp'].unique()

        rng = np.random.RandomState(seed)  # Initialize RandomState with the seed
        rng.shuffle(unique_base_sim_ids)  # Shuffle in place

        num_val_base_ids = int(len(unique_base_sim_ids) * val_ratio)
        # Ensure at least one validation base ID if val_ratio > 0 and there are enough unique base IDs
        if num_val_base_ids == 0 and val_ratio > 0 and len(unique_base_sim_ids) > 0:
            num_val_base_ids = 1

        val_base_sim_ids = unique_base_sim_ids[:num_val_base_ids]

        if seed is not None:
            print(f"Splitting by base_sim_id (delay variant grouping active, using seed {seed}).")
            print(f"Total unique base_sim_ids: {len(unique_base_sim_ids)}")
            print(f"Number of base_sim_ids for validation: {num_val_base_ids}")
            print(f"Validation base_sim_ids: {val_base_sim_ids.tolist()}")

        df_val = df_spatial[df_spatial['_base_sim_id_temp'].isin(val_base_sim_ids)].copy()
        df_train = df_spatial[~df_spatial['_base_sim_id_temp'].isin(val_base_sim_ids)].copy()

        # Clean up the temporary column
        df_train = df_train.drop(columns=['_base_sim_id_temp'])
        df_val = df_val.drop(columns=['_base_sim_id_temp'])

        print(f"Split by base_sim_id: {len(df_train)} train rows (from {df_train['sim_id'].nunique()} unique sim_ids), "
              f"{len(df_val)} val rows (from {df_val['sim_id'].nunique()} unique sim_ids).")
    else:
        # Original split by unique sim_id
        unique_sim_ids = df_spatial['sim_id'].unique()

        rng = np.random.RandomState(seed)  # Initialize RandomState with the seed
        rng.shuffle(unique_sim_ids)  # Shuffle in place

        num_val_ids = int(len(unique_sim_ids) * val_ratio)
        # Ensure at least one validation ID if val_ratio > 0 and there are enough unique IDs
        if num_val_ids == 0 and val_ratio > 0 and len(unique_sim_ids) > 0:
            num_val_ids = 1

        val_sim_ids = unique_sim_ids[:num_val_ids]

        if seed is not None:  # Log if a seed is used for easier debugging/verification
            print(f"Splitting by sim_id (using seed {seed}).")
            print(f"Total unique sim_ids: {len(unique_sim_ids)}")
            print(f"Number of sim_ids for validation: {num_val_ids}")
            print(f"Validation sim_ids: {val_sim_ids.tolist()}")

        df_val = df_spatial[df_spatial['sim_id'].isin(val_sim_ids)].copy()
        df_train = df_spatial[~df_spatial['sim_id'].isin(val_sim_ids)].copy()
        print(f"Split by sim_id: {len(df_train)} train rows, {len(df_val)} val rows.")

    return df_train, df_val


def get_dataloaders(spatiotemp_path, ecg_path, batch_size=128, val_ratio=0.2, seed=None, group_by_delay_variant=False, device="cpu"):
    """
    1) Load spatial DataFrame from parquet (columns: x, y, z, T, sim_id)
    2) Load ECG dictionary from parquet.
    3) Split the DataFrame into train/val sets based on sim_id.
    4) Build Dataset objects that reference the ecg_dict.
    5) Return DataLoaders for training and validation.
    """
    num_workers = 1 if device == "cpu" else 4
    df_spatial = load_spatiotemp_df(spatiotemp_path)
    ecg_dict = load_ecg_dict(ecg_path)
    # df_spatial = df_spatial[df_spatial["sim_id"] >= 20000]

    df_train, df_val = split_train_val_df(df_spatial, val_ratio=val_ratio,
                                          random_sample=False, seed=seed, group_by_delay_variant=group_by_delay_variant)

    train_dataset = SpatiotemporalECGDataset(df_train, ecg_dict)
    val_dataset = SpatiotemporalECGDataset(df_val, ecg_dict)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=num_workers,
                            pin_memory=True)
    '''
    test_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=4,
                        pin_memory=True)

    '''
    return train_loader, val_loader
