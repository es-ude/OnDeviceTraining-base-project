"""Vergleicht mlp_mnist_stress_host gegen PyTorch-Referenz (5-hidden-layer MLP)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare import Config, main_cli

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist():
    x_train = np.load(DATA_DIR / "mnist_train_x.npy").astype(np.float32).reshape(-1, 784)
    y_train = np.load(DATA_DIR / "mnist_train_y.npy").astype(np.int64)
    x_test  = np.load(DATA_DIR / "mnist_test_x.npy").astype(np.float32).reshape(-1, 784)
    y_test  = np.load(DATA_DIR / "mnist_test_y.npy").astype(np.int64)
    if y_train.ndim == 2:
        y_train = y_train.argmax(axis=1)
        y_test  = y_test.argmax(axis=1)
    return x_train, y_train, x_test, y_test


def build_model():
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 32),   nn.ReLU(),
        nn.Linear(32, 10),
    )
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def train(model, seed):
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = load_mnist()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(5):  # matcht NUM_EPOCHS im C-Code
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            probs = F.softmax(model(xb), dim=1)
            loss = F.nll_loss(torch.log(probs + 1e-12), yb)
            loss.backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(x_test)).argmax(dim=1)
        acc = (preds == torch.from_numpy(y_test)).float().mean().item() * 100.0
    return acc


if __name__ == "__main__":
    cfg = Config(
        binary_path="build/HOST-Debug/HOST",
        example_name="mlp_mnist_stress_host",
        stdout_metric_regex=r"accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=5,
        sigma_multiplier=2.0,
        pytorch_build=build_model,
        pytorch_train=train,
    )
    main_cli(cfg)
