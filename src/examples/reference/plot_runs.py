"""Visualisiert ODT vs PyTorch Seed-Sweep-Ergebnisse aus runs/.

Ausgabe: PNGs unter plots/. Aufruf (aus Repo-Root):

    uv run src/examples/reference/plot_runs.py

Erzeugte Plots
- 01_training_curves_{example}.png : Loss + Accuracy vs Epoche, Framework = Farbe,
  Phase (train/eval) = Linienart, Band = ±1 sd über alle Seeds je Epoche
- 02_class_distributions.png : Klassen-Histogramme für Full-MNIST (host/stress_host)
  und MCU-Compile-Time-Subset (100/20). Subset ist seed-invariant.
- 03a_final_accuracy_per_seed.png : Box + Strip je (example, framework), macht die
  bimodal-Streuung von stress_host visuell sofort sichtbar.
- 03b_framework_gap_per_epoch.png : (PyTorch mean) - (ODT mean) pro Epoche je
  Example — zeigt, wann der Gap entsteht und ob er konvergiert oder driftet.
- 03c_seed_paired_scatter.png : Für gleichen Bookkeeping-Seed-Index: ODT-final
  vs PyTorch-final. Punkte weit unter der Diagonalen = dieser Seed konvergiert
  nur bei PyTorch.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RUNS = Path("runs")
DATA = Path("src/examples/data")
OUT = Path("src/examples/plots")

EXAMPLES = [
    "mlp_mnist_float32_host",
    "mlp_mnist_stress_host",
    "mlp_mnist_float32_mcu",
]

FRAMEWORK_PALETTE = {"odt": "#E76F51", "pytorch": "#2A9D8F"}

CSV_NAME_RE = re.compile(
    r"^(?P<example>.+)_(?P<framework>odt|pytorch)_seed(?P<seed>\d+)\.csv$"
)


def load_runs() -> pd.DataFrame:
    rows = []
    for csv in sorted(RUNS.glob("*.csv")):
        m = CSV_NAME_RE.match(csv.name)
        if not m or m.group("example") not in EXAMPLES:
            continue
        df = pd.read_csv(csv)
        df["example"] = m.group("example")
        df["framework"] = m.group("framework")
        df["seed"] = int(m.group("seed"))
        rows.append(df)
    if not rows:
        raise RuntimeError(f"no matching CSVs under {RUNS.resolve()}")
    return pd.concat(rows, ignore_index=True)


def plot_training_curves(df: pd.DataFrame) -> None:
    for ex in EXAMPLES:
        sub = df[df["example"] == ex].copy()
        loss_df = sub.melt(
            id_vars=["epoch", "framework", "seed"],
            value_vars=["train_loss", "eval_loss"],
            var_name="phase",
            value_name="loss",
        )
        loss_df["phase"] = loss_df["phase"].str.replace("_loss", "", regex=False)

        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(16, 6))
        sns.lineplot(
            data=loss_df, x="epoch", y="loss",
            hue="framework", style="phase",
            estimator="mean", errorbar="sd",
            palette=FRAMEWORK_PALETTE, ax=ax_l,
        )
        ax_l.set_yscale("log")
        ax_l.set_title(f"{ex} — Loss (mean ± sd)")
        ax_l.set_ylabel("loss (log)")

        sns.lineplot(
            data=sub, x="epoch", y="test_accuracy",
            hue="framework", estimator="mean", errorbar="sd",
            palette=FRAMEWORK_PALETTE, ax=ax_a,
        )
        ax_a.set_title(f"{ex} — Test accuracy (mean ± sd)")
        ax_a.set_ylabel("accuracy (%)")

        n_odt = sub[sub["framework"] == "odt"]["seed"].nunique()
        n_pt = sub[sub["framework"] == "pytorch"]["seed"].nunique()
        fig.suptitle(f"{ex}  (N_odt={n_odt}, N_pt={n_pt})")
        fig.tight_layout()
        path = OUT / f"01_training_curves_{ex}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


def _parse_c_label_array(header: Path) -> np.ndarray:
    text = header.read_text()
    m = re.search(r"_labels\[[^\]]*\]\s*=\s*\{([^}]+)\}", text)
    if not m:
        raise RuntimeError(f"no `_labels[...] = {{...}}` in {header}")
    return np.array([int(v) for v in re.findall(r"\d+", m.group(1))], dtype=np.uint8)


def _full_mnist_labels(path: Path) -> np.ndarray:
    y = np.load(path)
    if y.ndim == 2 and y.shape[1] == 10:
        y = y.argmax(axis=1)
    return y.astype(np.uint8)


def plot_class_distributions() -> None:
    y_train_full = _full_mnist_labels(DATA / "mnist_train_y.npy")
    y_test_full = _full_mnist_labels(DATA / "mnist_test_y.npy")
    y_train_mcu = _parse_c_label_array(DATA / "mnist_train_subset.h")
    y_test_mcu = _parse_c_label_array(DATA / "mnist_test_subset.h")

    datasets = [
        (f"Full MNIST — train ({y_train_full.size})", y_train_full),
        (f"Full MNIST — test ({y_test_full.size})", y_test_full),
        (f"MCU subset — train ({y_train_mcu.size})", y_train_mcu),
        (f"MCU subset — test ({y_test_mcu.size})", y_test_mcu),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (title, labels) in zip(axes.flat, datasets):
        counts = pd.Series(labels).value_counts().reindex(range(10), fill_value=0)
        sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#264653")
        ax.set_title(title)
        ax.set_xlabel("class")
        ax.set_ylabel("count")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)
    fig.suptitle(
        "Class distributions — MCU-Subset ist seed-invariant "
        "(dieselben 100 train / 20 test über alle Seeds)"
    )
    fig.tight_layout()
    path = OUT / "02_class_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def _finals(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("epoch").groupby(["example", "framework", "seed"]).tail(1)


def plot_final_accuracy_per_seed(df: pd.DataFrame) -> None:
    final = _finals(df)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for ax, ex in zip(axes, EXAMPLES):
        sub = final[final["example"] == ex]
        sns.boxplot(
            data=sub, x="framework", y="test_accuracy", hue="framework",
            palette=FRAMEWORK_PALETTE, ax=ax, showmeans=True, legend=False,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=8),
        )
        sns.stripplot(
            data=sub, x="framework", y="test_accuracy",
            color="black", size=5, alpha=0.6, ax=ax,
        )
        n_o = sub[sub["framework"] == "odt"]["seed"].nunique()
        n_p = sub[sub["framework"] == "pytorch"]["seed"].nunique()
        ax.set_title(f"{ex}\n(N_odt={n_o}, N_pt={n_p})")
        ax.set_ylabel("final test accuracy (%)")
        ax.set_xlabel("")
    fig.suptitle("Final test accuracy per seed — Box = IQR, Diamond = mean")
    fig.tight_layout()
    path = OUT / "03a_final_accuracy_per_seed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_framework_gap_per_epoch(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["example", "framework", "epoch"])["test_accuracy"]
        .mean()
        .unstack("framework")
    )
    agg["gap_pp"] = agg["pytorch"] - agg["odt"]
    agg = agg.reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg, x="epoch", y="gap_pp", hue="example", marker="o", ax=ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("PyTorch mean − ODT mean  (pp)")
    ax.set_title("Framework-Gap pro Epoche (positiv = PyTorch ist vorn)")
    fig.tight_layout()
    path = OUT / "03b_framework_gap_per_epoch.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_host_per_seed(df: pd.DataFrame) -> None:
    """Diagnostic: alle HOST-Seeds individuell, nicht aggregiert.

    Pro Seed zwei Axes nebeneinander: links Loss (train=solid, eval=dashed,
    log-y), rechts Test-Accuracy. ODT=rot, PyTorch=teal aus FRAMEWORK_PALETTE.
    Zweck: sichtbar machen, ob einzelne Runs qualitativ abweichen (z.B. stuck
    seeds, divergente Curves) — Aggregate-Plots können sowas mitteln-weg.
    """
    ex = "mlp_mnist_float32_host"
    sub = df[df["example"] == ex].copy()
    seeds = sorted(sub["seed"].unique())
    n_cols_seed = 4
    n_rows = (len(seeds) + n_cols_seed - 1) // n_cols_seed
    fig, axes = plt.subplots(
        n_rows, n_cols_seed * 2,
        figsize=(n_cols_seed * 2 * 3.5, n_rows * 2.8),
        squeeze=False,
    )
    for idx, seed in enumerate(seeds):
        r = idx // n_cols_seed
        c = (idx % n_cols_seed) * 2
        ax_l, ax_a = axes[r, c], axes[r, c + 1]
        for fw in ("odt", "pytorch"):
            fw_df = sub[(sub["seed"] == seed) & (sub["framework"] == fw)]
            color = FRAMEWORK_PALETTE[fw]
            ax_l.plot(fw_df["epoch"], fw_df["train_loss"],
                      color=color, linestyle="-", linewidth=1.2,
                      label=f"{fw} train")
            ax_l.plot(fw_df["epoch"], fw_df["eval_loss"],
                      color=color, linestyle="--", linewidth=1.2,
                      label=f"{fw} eval")
            ax_a.plot(fw_df["epoch"], fw_df["test_accuracy"],
                      color=color, linestyle="-", linewidth=1.2, label=fw)
        ax_l.set_yscale("log")
        ax_l.set_title(f"seed {seed:02d} — loss", fontsize=10)
        ax_a.set_title(f"seed {seed:02d} — test acc", fontsize=10)
        ax_l.tick_params(labelsize=8)
        ax_a.tick_params(labelsize=8)
        ax_l.set_xlabel("")
        ax_a.set_xlabel("")
        if idx == 0:
            ax_l.legend(fontsize=7, loc="upper right")
            ax_a.legend(fontsize=7, loc="lower right")
    for j in range(len(seeds), n_rows * n_cols_seed):
        r = j // n_cols_seed
        c = (j % n_cols_seed) * 2
        axes[r, c].set_visible(False)
        axes[r, c + 1].set_visible(False)
    fig.suptitle(
        f"{ex} — Per-Seed individual curves "
        f"(solid=train, dashed=eval; ODT=red, PyTorch=teal)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    path = OUT / "01b_host_per_seed.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_seed_paired_scatter(df: pd.DataFrame) -> None:
    final = _finals(df)
    wide = (
        final.pivot_table(
            index=["example", "seed"], columns="framework", values="test_accuracy"
        )
        .dropna()
        .reset_index()
    )

    g = sns.lmplot(
        data=wide, x="odt", y="pytorch", col="example", col_order=EXAMPLES,
        col_wrap=3, height=5,
        facet_kws={"sharex": False, "sharey": False},
        ci=None, scatter_kws={"s": 60, "alpha": 0.7},
    )
    for ax in g.axes.flat:
        mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
        mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([mn, mx], [mn, mx], "k:", linewidth=1, label="identity")
        ax.set_xlabel("ODT final accuracy (%)")
        ax.set_ylabel("PyTorch final accuracy (%)")
    g.figure.suptitle(
        "Per-seed pairing (Bookkeeping-Index) — Punkte über Diagonalen = PyTorch gewinnt",
        y=1.03,
    )
    path = OUT / "03c_seed_paired_scatter.png"
    g.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  wrote {path}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    OUT.mkdir(exist_ok=True)

    df = load_runs()
    summary = (
        df.groupby(["example", "framework"])["seed"].nunique().unstack("framework")
    )
    print("Loaded seed counts per (example, framework):")
    print(summary.to_string())
    print()

    plot_training_curves(df)
    plot_host_per_seed(df)
    plot_class_distributions()
    plot_final_accuracy_per_seed(df)
    plot_framework_gap_per_epoch(df)
    plot_seed_paired_scatter(df)


if __name__ == "__main__":
    main()
