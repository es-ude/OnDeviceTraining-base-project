"""Vergleicht mlp_mnist_float32_host gegen PyTorch-Referenz."""
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
    # PyTorch CE erwartet Klassen-Indices, aber MNIST labels aus .npy sind
    # one-hot (10-dim). Konvertieren:
    if y_train.ndim == 2 and y_train.shape[1] == 10:
        y_train = y_train.argmax(axis=1)
        y_test  = y_test.argmax(axis=1)
    return x_train, y_train, x_test, y_test


def build_model():
    model = nn.Sequential(
        nn.Linear(784, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )
    # Xavier uniform für weights, zero für bias — matcht ODT tensorInitWithDistribution(XAVIER_UNIFORM).
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def train(model, seed):
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = load_mnist()
    # ODT DataLoader shuffles ONCE at init (seed=42), never re-shuffles between epochs.
    # Replicate: shuffle the dataset once with seed=42, then iterate in fixed order.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    # shuffle=False because we already shuffled once above, matching ODT behaviour.
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, drop_last=True)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            # ODT nutzt Softmax + CrossEntropy-mit-Probs. Wir replizieren das
            # Exakt: log(softmax(logits)) dann NLL gegen Klassen-Index.
            probs = F.softmax(logits, dim=1)
            loss = F.nll_loss(torch.log(probs + 1e-12), yb)
            loss.backward()
            optim.step()

    # Test accuracy
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_test)
        y_t = torch.from_numpy(y_test)
        preds = model(x_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item() * 100.0
    return acc


if __name__ == "__main__":
    cfg = Config(
        binary_path="build/HOST-Debug/HOST",
        example_name="mlp_mnist_float32_host",
        stdout_metric_regex=r"accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=20,
        sigma_multiplier=2.0,
        pytorch_build=build_model,
        pytorch_train=train,
    )
    main_cli(cfg)
