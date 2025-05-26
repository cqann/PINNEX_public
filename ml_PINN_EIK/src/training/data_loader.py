import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class HeartDataset(Dataset):
    """
    A PyTorch Dataset for large spatiotemporal points (x, y, z, t) and targets (U, V).
    """

    def __init__(self, df):
        """
        df: a pandas DataFrame with columns ['x','y','z','t','U','V'] (float).
        """
        self.x = torch.tensor(df['x'].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(df['y'].values, dtype=torch.float32).view(-1, 1)
        self.z = torch.tensor(df['z'].values, dtype=torch.float32).view(-1, 1)
        self.t = torch.tensor(df['t'].values, dtype=torch.float32).view(-1, 1)
        self.U = torch.tensor(df['U'].values, dtype=torch.float32).view(-1, 1)
        self.V = torch.tensor(df['V'].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single sample at 'idx'
        return (self.x[idx],
                self.y[idx],
                self.z[idx],
                self.t[idx],
                self.U[idx],
                self.V[idx])


def load_heart_data(file_path):
    """
    Reads a parquet file (or CSV) and returns a DataFrame for easy splitting, etc.
    """
    df = pd.read_parquet(file_path)
    df = df.reset_index()
    return df


def split_train_val(df, val_ratio=0.2, shuffle=True):
    """
    Splits the dataframe into train/val subsets.
    """
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = int(len(df) * val_ratio)
    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]
    return df_train, df_val


def get_dataloaders(path, batch_size=128, val_ratio=0.2):
    """
    Returns train_loader, val_loader given a path to data.
    """
    # 1) Load DF
    df = load_heart_data(path)

    # 2) Split
    df_train, df_val = split_train_val(df, val_ratio=val_ratio)

    # 3) Create Datasets
    train_dataset = HeartDataset(df_train)
    val_dataset = HeartDataset(df_val)

    # 4) Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
