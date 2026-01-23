from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def autoencoder_scores(
    X_train_df: pd.DataFrame,
    X_val_df: Optional[pd.DataFrame],
    X_test_df: Optional[pd.DataFrame],
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    random_state: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    input_dim = X_train_df.shape[1]
    model = Autoencoder(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch).mean()
            loss.backward()
            optimizer.step()

    def compute_scores(df: pd.DataFrame) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            data = torch.tensor(df.values, dtype=torch.float32).to(device)
            recon = model(data)
            per_row = loss_fn(recon, data).mean(dim=1)
            return per_row.detach().cpu().numpy()

    train_scores = compute_scores(X_train_df)
    val_scores = compute_scores(X_val_df) if X_val_df is not None and not X_val_df.empty else None
    test_scores = compute_scores(X_test_df) if X_test_df is not None and not X_test_df.empty else None

    return train_scores, val_scores, test_scores
