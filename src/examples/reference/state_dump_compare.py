"""Vergleicht ODT- und PyTorch-State-Dumps numerisch (Plan 2, Tasks 2-6).

Erwartet zwei Verzeichnisse mit .npy-Dateien, geschrieben von:
  - C-Binary im ODT_SINGLE_BATCH=1-Modus (mlp_mnist_depth_sweep_host)
  - depth_sweep_ref.py --state-dump <dir>

Vergleicht fuer jede gemeinsame .npy-Datei: max |abs diff|, max |rel diff|.

Hypothesen-Schalter:
  --loss-variant {sum,mean}: welche PyTorch-Loss-Variante fuer Vergleich.
  --scale-odt-grad-by <float>: ODT-Grads pre-multipliziert, dann verglichen --
    testet "was waere wenn ODT dividiert haette". Typisch: batch_size^-1 = 0.03125.

Filenames (adaptiert fuer Task 1's tatsaechliche Dumps):
  ODT:  pre_{w,b}_{k}.npy, post_grad_{w,b}_{k}.npy, loss_sum.npy, loss_mean.npy.
  PT:   pre_{w,b}_{k}.npy, post_grad_{w,b}_{sum,mean}_{k}.npy,
        loss_sum.npy, loss_mean.npy, pre_relu_{k}.npy, softmax_out.npy.

Fuer loss laedt das Skript auf ODT-Seite `loss_{variant}.npy` (beide Varianten
liegen vor), faellt auf `loss.npy` zurueck, falls nur die Legacy-Variante
existiert.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load(dir_: Path, name: str):
    p = dir_ / f"{name}.npy"
    return np.load(p) if p.exists() else None


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}"}
    diff = a - b
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom > 1e-12, denom, 1.0)
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_rel_diff": float(np.max(np.abs(diff) / denom)),
        "shape": a.shape,
    }


def compare(odt: Path, pt: Path, loss_variant: str, scale_odt_grad_by: float,
            num_linear: int):
    print(f"\n=== Compare ODT:{odt}  vs  PyTorch:{pt} ===")
    print(f"    loss_variant={loss_variant}  scale_odt_grad_by={scale_odt_grad_by}")
    # Initial weights.
    for k in range(num_linear):
        for kind in ("w", "b"):
            a = load(odt, f"pre_{kind}_{k}")
            b = load(pt, f"pre_{kind}_{k}")
            if a is None or b is None:
                continue
            print(f"  pre_{kind}_{k}: {stats(a, b)}")

    # Post-activations (if dumped).
    for k in range(num_linear):
        a = load(odt, f"pre_relu_{k}")
        b = load(pt, f"pre_relu_{k}")
        if a is None or b is None:
            continue
        print(f"  pre_relu_{k}: {stats(a, b)}")
    a = load(odt, "softmax_out")
    b = load(pt, "softmax_out")
    if a is not None and b is not None:
        print(f"  softmax_out: {stats(a, b)}")

    # Loss. On ODT side both `loss_sum.npy` and `loss_mean.npy` are produced
    # by depth_sweep_host's dump-mode; fall back to a bare `loss.npy` if that
    # is all that exists (older dumps).
    a = load(odt, f"loss_{loss_variant}")
    if a is None:
        a = load(odt, "loss")
    b = load(pt, f"loss_{loss_variant}")
    if a is not None and b is not None:
        print(f"  loss (odt.{loss_variant} vs pt.{loss_variant}): {stats(a, b)}")

    # Gradients.
    for k in range(num_linear):
        for kind in ("w", "b"):
            a = load(odt, f"post_grad_{kind}_{k}")
            b = load(pt, f"post_grad_{kind}_{loss_variant}_{k}")
            if a is None or b is None:
                continue
            a_scaled = a * scale_odt_grad_by
            print(f"  post_grad_{kind}_{k} (odt*{scale_odt_grad_by}): {stats(a_scaled, b)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--odt-dir", type=Path, required=True)
    p.add_argument("--pt-dir", type=Path, required=True)
    p.add_argument("--loss-variant", choices=["sum", "mean"], default="sum")
    p.add_argument("--scale-odt-grad-by", type=float, default=1.0)
    p.add_argument("--num-linear", type=int, required=True,
                   help="Number of Linear layers in the model (1 + hidden_layers).")
    args = p.parse_args()
    compare(args.odt_dir, args.pt_dir, args.loss_variant, args.scale_odt_grad_by,
            args.num_linear)
