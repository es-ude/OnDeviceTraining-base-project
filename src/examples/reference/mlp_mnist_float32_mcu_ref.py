"""Vergleicht mlp_mnist_float32_mcu gegen PyTorch-Referenz (100-sample subset)."""
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

TRAIN_SUBSET_SIZE = 100
TEST_SUBSET_SIZE = 20


def load_mnist_subset():
    x_train = np.load(DATA_DIR / "mnist_train_x.npy").astype(np.float32).reshape(-1, 784)[:TRAIN_SUBSET_SIZE]
    y_train = np.load(DATA_DIR / "mnist_train_y.npy").astype(np.int64)[:TRAIN_SUBSET_SIZE]
    x_test  = np.load(DATA_DIR / "mnist_test_x.npy").astype(np.float32).reshape(-1, 784)[:TEST_SUBSET_SIZE]
    y_test  = np.load(DATA_DIR / "mnist_test_y.npy").astype(np.int64)[:TEST_SUBSET_SIZE]
    if y_train.ndim == 2:
        y_train = y_train.argmax(axis=1)
        y_test  = y_test.argmax(axis=1)
    return x_train, y_train, x_test, y_test


def build_model():
    model = nn.Sequential(nn.Linear(784, 20), nn.ReLU(), nn.Linear(20, 10))
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def train(model, seed):
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = load_mnist_subset()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    optim = torch.optim.SGD(model.parameters(), lr=0.01)  # ODT mcu nutzt 0.01

    for epoch in range(3):
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
        example_name="mlp_mnist_float32_mcu",
        stdout_metric_regex=r"subset_accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=5,
        sigma_multiplier=2.0,
        pytorch_build=build_model,
        pytorch_train=train,
    )
    main_cli(cfg)
