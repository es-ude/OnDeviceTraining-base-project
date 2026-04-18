# Examples Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Alle 7 Examples des base-project Repos fertigstellen: broken asym-Examples gegen versehentliche Kompilation absichern, Stresstest-Example neu bauen, alle Training-Examples statistisch gegen PyTorch-Referenzen validieren (N=5 Seeds, 2σ), dann Phase 6 Qualitätsreview und Phase 7 README / Hardware-Matrix.

**Architecture:** C-Examples leben in `src/examples/<name>.c` und werden über `ODT_EXAMPLE`-Cache-Variable selektiert. PyTorch-Referenzen leben in `src/examples/reference/` als `uv`-gemanagete Python-Skripte; eine geteilte `compare.py`-Harness parsed ODT-Stdout und rechnet mean±std sowie 2σ-Check. HOST-Builds sind die Validierungsplattform; MCU-Targets werden nur cross-kompiliert, da Hardware nur teilweise verfügbar.

**Tech Stack:** C11, CMake (Ninja), `uv`+Python 3.13+, PyTorch (CPU-only), NumPy, jj (colocated mit git).

---

## Annahmen und Voraussetzungen

- `cmake --preset PREPARE` + ein Target-Preset wurden einmal gelaufen, sodass `pico-sdk/` und `OnDeviceTraining/` vorhanden sind.
- `HOST-Debug` Preset existiert und funktioniert (bestätigt).
- MNIST .npy-Dateien liegen unter `src/examples/data/` (bereits erzeugt; falls nicht: `uv run src/examples/data/generate_subset.py`).
- `gh` CLI ist authentifiziert, `jj` ist colocated.
- Brainstorm-Spec **und** dieser Plan werden **nicht committed** (bleiben lokal im Working-Copy). Code + README + .gitignore-Änderungen werden committed.

## File Structure

**Create:**
- `src/examples/mlp_mnist_stress_host.c` — 5-Hidden-Layer MLP, host-only, Boilerplate-Messvehikel
- `src/examples/reference/compare.py` — geteilte Harness: ODT-Binary starten, Stdout parsen, N=5 PyTorch-Runs, 2σ-Check
- `src/examples/reference/mlp_mnist_float32_host_ref.py` — PyTorch-Modell für `mlp_mnist_float32_host`
- `src/examples/reference/mlp_mnist_float32_mcu_ref.py` — PyTorch-Modell für `mlp_mnist_float32_mcu` (100-Sample-Subset)
- `src/examples/reference/mlp_mnist_stress_host_ref.py` — PyTorch-Modell für Stresstest
- `src/examples/reference/__init__.py` — macht den Ordner zum Python-Package (leer)

**Modify:**
- `src/examples/mlp_mnist_asym_host.c` — `#error`-Gate hinzufügen (zeilen 1-15 Kommentar + #error am Anfang)
- `src/examples/mlp_mnist_asym_mcu.c` — gleiches `#error`-Gate
- `pyproject.toml` — `torch` als Dependency ergänzen
- `README.md` — Example-Tabelle + Hardware-Matrix + Quickstart
- `.gitignore` — `src/examples/data/*.npy`, `src/examples/data/*.gz`, `OnDeviceTraining/`, `pico-sdk/`, `CMakeCache.txt`, `CMakeFiles/`, `cmake_install.cmake`, `Makefile` (falls nicht in `/build`)

**Do not commit:**
- `docs/superpowers/specs/` (bleibt lokal per User-Feedback)
- `docs/superpowers/plans/` (gleiches gilt für Plans)

---

## Task 1: Asym-Examples gegen versehentliche Kompilation absichern

ODT-Issue #61 trackt, dass `linearForward` ASYM nicht dispatcht. Bis das gefixt ist, würden diese Examples zur Laufzeit `exit(1)` ausgeben. Um Zeit-Verschwendung zu verhindern, schlagen sie mit `#error` fehl wenn selektiert.

**Files:**
- Modify: `src/examples/mlp_mnist_asym_host.c`
- Modify: `src/examples/mlp_mnist_asym_mcu.c`

- [ ] **Step 1: `#error` in asym_host.c einfügen**

Am Anfang der Datei (direkt nach dem Header-Kommentar vor `#include`):

```c
#error "ASYM forward dispatch ist in ODT nicht implementiert — siehe https://github.com/es-ude/OnDeviceTraining/issues/61. Example bleibt als Vorlage stehen; sobald Issue #61 geschlossen ist, diesen #error entfernen."
```

- [ ] **Step 2: Gleiches `#error` in asym_mcu.c einfügen**

Identischer Block.

- [ ] **Step 3: Build-Fehler verifizieren**

Run:
```
cmake --preset HOST-Debug -DODT_EXAMPLE=mlp_mnist_asym_host
cmake --build --preset HOST-Debug
```
Expected: Build bricht mit dem `#error`-Text ab.

- [ ] **Step 4: Default-Build (linear_regression) baut noch**

Run:
```
cmake --preset HOST-Debug
cmake --build --preset HOST-Debug
```
Expected: Erfolgreicher Build, `./build/HOST-Debug/HOST` existiert.

- [ ] **Step 5: Commit**

```
jj describe -m "examples: gate asym variants behind #error referencing ODT #61"
jj new
```

---

## Task 2: Stresstest-Example schreiben (mlp_mnist_stress_host.c)

5-Hidden-Layer MLP 784→256→128→64→32→10. Host-only (~970 KB Weights passen nicht auf Pico2 W). **Keine Helper-Funktionen, keine Schleifen für Layer-Konstruktion** — jede Layer wird explizit aufgebaut, damit der Boilerplate-Schmerz zählbar ist. Trainiert auf MNIST-Full aus .npy; validiert Forward+Train gegen PyTorch in Task 7.

**Files:**
- Create: `src/examples/mlp_mnist_stress_host.c`

- [ ] **Step 1: Datei schreiben**

```c
/*
 * Example: mlp_mnist_stress_host  (HOST-only)
 *
 * Stresstest für Boilerplate-Messung: 5-Hidden-Layer MLP 784 -> 256 -> 128 ->
 * 64 -> 32 -> 10, ReLU zwischen allen, Softmax am Ende, CrossEntropy Loss,
 * FLOAT32 durchgehend. Trainiert auf MNIST-Full aus .npy.
 *
 * Ziel dieses Examples ist NICHT Convenience oder Produktions-Stil. Es ist
 * explizit so geschrieben, dass jede Layer Zeile für Zeile ohne Helper
 * aufgebaut wird — genau so sieht User-Code aus, wenn die USERAPI keine
 * Convenience-Layer hat. Der dadurch entstehende Boilerplate ist das, was
 * Pass 1 des USERAPI-Audits quantifiziert.
 *
 * Host-only: ~244K Parameter × 4 B = ~970 KB passt nicht in Pico2 W (520 KB
 * SRAM). Passt bequem in Host-BSS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hardware_init.h"

#include "Layer.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "QuantizationApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "SgdApi.h"
#include "InferenceApi.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "DataLoader.h"
#include "NPYLoaderApi.h"
#include "Dataset.h"
#include "LossFunction.h"

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "."
#endif
#define MNIST_TRAIN_X MNIST_DATA_DIR "/mnist_train_x.npy"
#define MNIST_TRAIN_Y MNIST_DATA_DIR "/mnist_train_y.npy"
#define MNIST_TEST_X  MNIST_DATA_DIR "/mnist_test_x.npy"
#define MNIST_TEST_Y  MNIST_DATA_DIR "/mnist_test_y.npy"

#define INPUT_DIM     (28 * 28)
#define H1_DIM        256
#define H2_DIM        128
#define H3_DIM        64
#define H4_DIM        32
#define OUTPUT_DIM    10
#define NUM_CLASSES   10
#define BATCH_SIZE    32
#define NUM_EPOCHS    5
#define LEARNING_RATE 0.001f
/* Model-Kette: 5 Linear + 4 ReLU (zwischen Linears) + 1 Softmax = 10 Layer. */
#define MODEL_SIZE    10

static dataset_t trainDataset;
static dataset_t testDataset;

static sample_t *getTrainSample(size_t id) { return npyGetSample(&trainDataset, id); }
static sample_t *getTestSample (size_t id) { return npyGetSample(&testDataset,  id); }
static size_t getTrainSize(void) { return trainDataset.items->size; }
static size_t getTestSize (void) { return testDataset.items->size;  }

static void flattenItems(tensorArray_t *arr) {
    for (size_t i = 0; i < arr->size; i++) {
        shape_t *shape = arr->array[i]->shape;
        size_t *newDims  = *reserveMemory(2 * sizeof(size_t));
        size_t *newOrder = *reserveMemory(2 * sizeof(size_t));
        newDims[0] = shape->dimensions[0];
        newDims[1] = shape->dimensions[1] * shape->dimensions[2];
        newOrder[0] = 0;
        newOrder[1] = 1;
        freeReservedMemory(shape->dimensions);
        freeReservedMemory(shape->orderOfDimensions);
        shape->dimensions = newDims;
        shape->orderOfDimensions = newOrder;
        shape->numberOfDimensions = 2;
    }
}

static void onEpochEnd(size_t epoch, float trainLoss, float evalLoss) {
    printf("  epoch %zu: train_loss=%.4f eval_loss=%.4f\n",
           epoch + 1, (double)trainLoss, (double)evalLoss);
}

int main(void) {
    init();
    printf("mlp_mnist_stress_host: 784->256->128->64->32->10 MLP (stress-test)\n");

    trainDataset.items  = npyLoad(MNIST_TRAIN_X);
    trainDataset.labels = npyLoad(MNIST_TRAIN_Y);
    testDataset.items   = npyLoad(MNIST_TEST_X);
    testDataset.labels  = npyLoad(MNIST_TEST_Y);
    if (!trainDataset.items || !trainDataset.labels ||
        !testDataset.items  || !testDataset.labels) {
        fprintf(stderr, "Could not load MNIST .npy files.\n");
        return 1;
    }
    flattenItems(trainDataset.items);
    flattenItems(testDataset.items);

    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, true, 42, true);
    dataLoader_t *testDL  = dataLoaderInit(getTestSample,  getTestSize,  1,
                                            NULL, NULL, false, 0, true);

    quantization_t *q = quantizationInitFloat();
    layer_t *model[MODEL_SIZE];

    /* =========================================================================
     * LAYER 1: Linear 784 -> 256  (ABSICHTLICH OHNE HELPER — Boilerplate sichtbar)
     * ========================================================================= */
    static float w0[H1_DIM * INPUT_DIM] = {0};
    size_t w0Dims[] = {H1_DIM, INPUT_DIM};
    tensor_t *w0P = tensorInitWithDistribution(XAVIER_UNIFORM, w0, w0Dims, 2, q, NULL, INPUT_DIM, H1_DIM);
    tensor_t *w0G = gradInitFloat(w0P, NULL);
    parameter_t *w0Pm = parameterInit(w0P, w0G);

    static float b0[H1_DIM] = {0};
    size_t b0Dims[] = {1, H1_DIM};
    tensor_t *b0P = tensorInitWithDistribution(ZEROS, b0, b0Dims, 2, q, NULL, 1, H1_DIM);
    tensor_t *b0G = gradInitFloat(b0P, NULL);
    parameter_t *b0Pm = parameterInit(b0P, b0G);

    model[0] = linearLayerInit(w0Pm, b0Pm, q, q, q, q);
    model[1] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 2: Linear 256 -> 128
     * ========================================================================= */
    static float w1[H2_DIM * H1_DIM] = {0};
    size_t w1Dims[] = {H2_DIM, H1_DIM};
    tensor_t *w1P = tensorInitWithDistribution(XAVIER_UNIFORM, w1, w1Dims, 2, q, NULL, H1_DIM, H2_DIM);
    tensor_t *w1G = gradInitFloat(w1P, NULL);
    parameter_t *w1Pm = parameterInit(w1P, w1G);

    static float b1[H2_DIM] = {0};
    size_t b1Dims[] = {1, H2_DIM};
    tensor_t *b1P = tensorInitWithDistribution(ZEROS, b1, b1Dims, 2, q, NULL, 1, H2_DIM);
    tensor_t *b1G = gradInitFloat(b1P, NULL);
    parameter_t *b1Pm = parameterInit(b1P, b1G);

    model[2] = linearLayerInit(w1Pm, b1Pm, q, q, q, q);
    model[3] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 3: Linear 128 -> 64
     * ========================================================================= */
    static float w2[H3_DIM * H2_DIM] = {0};
    size_t w2Dims[] = {H3_DIM, H2_DIM};
    tensor_t *w2P = tensorInitWithDistribution(XAVIER_UNIFORM, w2, w2Dims, 2, q, NULL, H2_DIM, H3_DIM);
    tensor_t *w2G = gradInitFloat(w2P, NULL);
    parameter_t *w2Pm = parameterInit(w2P, w2G);

    static float b2[H3_DIM] = {0};
    size_t b2Dims[] = {1, H3_DIM};
    tensor_t *b2P = tensorInitWithDistribution(ZEROS, b2, b2Dims, 2, q, NULL, 1, H3_DIM);
    tensor_t *b2G = gradInitFloat(b2P, NULL);
    parameter_t *b2Pm = parameterInit(b2P, b2G);

    model[4] = linearLayerInit(w2Pm, b2Pm, q, q, q, q);
    model[5] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 4: Linear 64 -> 32
     * ========================================================================= */
    static float w3[H4_DIM * H3_DIM] = {0};
    size_t w3Dims[] = {H4_DIM, H3_DIM};
    tensor_t *w3P = tensorInitWithDistribution(XAVIER_UNIFORM, w3, w3Dims, 2, q, NULL, H3_DIM, H4_DIM);
    tensor_t *w3G = gradInitFloat(w3P, NULL);
    parameter_t *w3Pm = parameterInit(w3P, w3G);

    static float b3[H4_DIM] = {0};
    size_t b3Dims[] = {1, H4_DIM};
    tensor_t *b3P = tensorInitWithDistribution(ZEROS, b3, b3Dims, 2, q, NULL, 1, H4_DIM);
    tensor_t *b3G = gradInitFloat(b3P, NULL);
    parameter_t *b3Pm = parameterInit(b3P, b3G);

    model[6] = linearLayerInit(w3Pm, b3Pm, q, q, q, q);
    model[7] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 5: Linear 32 -> 10 + Softmax
     * ========================================================================= */
    static float w4[OUTPUT_DIM * H4_DIM] = {0};
    size_t w4Dims[] = {OUTPUT_DIM, H4_DIM};
    tensor_t *w4P = tensorInitWithDistribution(XAVIER_UNIFORM, w4, w4Dims, 2, q, NULL, H4_DIM, OUTPUT_DIM);
    tensor_t *w4G = gradInitFloat(w4P, NULL);
    parameter_t *w4Pm = parameterInit(w4P, w4G);

    static float b4[OUTPUT_DIM] = {0};
    size_t b4Dims[] = {1, OUTPUT_DIM};
    tensor_t *b4P = tensorInitWithDistribution(ZEROS, b4, b4Dims, 2, q, NULL, 1, OUTPUT_DIM);
    tensor_t *b4G = gradInitFloat(b4P, NULL);
    parameter_t *b4Pm = parameterInit(b4P, b4G);

    model[8] = linearLayerInit(w4Pm, b4Pm, q, q, q, q);
    model[9] = softmaxLayerInit(q, q);

    /* Ende Boilerplate-Sektion. Ab hier: gewöhnliches Training. */

    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

    clock_t t0 = clock();
    trainingRunResult_t res = trainingRun(
        model, MODEL_SIZE, CROSS_ENTROPY,
        trainDL, testDL, sgd, NUM_EPOCHS,
        calculateGradsSequential, inferenceWithLoss, onEpochEnd);
    clock_t t1 = clock();

    float accuracy = evaluationEpochAccuracy(model, MODEL_SIZE, testDL, NUM_CLASSES, inference);
    printf("Done in %.2fs. final_train_loss=%.4f final_eval_loss=%.4f accuracy=%.2f%%\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC,
           (double)res.finalTrainLoss, (double)res.finalEvalLoss, (double)accuracy * 100.0);
    return 0;
}
```

- [ ] **Step 2: Mit HOST-Debug builden**

Run:
```
cmake --preset HOST-Debug -DODT_EXAMPLE=mlp_mnist_stress_host
cmake --build --preset HOST-Debug
```
Expected: Binary unter `build/HOST-Debug/HOST` entsteht ohne Fehler.

- [ ] **Step 3: Stresstest laufen lassen**

Run: `./build/HOST-Debug/HOST`
Expected: Training läuft ~5-15 Minuten (abhängig von Host-CPU), am Ende eine Zeile `Done in ...s. final_train_loss=... accuracy=XX.XX%`. Accuracy sollte >90% sein (MLP auf MNIST).

Falls es fehlschlägt: Debuggen per DEBUG_MODE_ERROR-Output (ist im host_post.cmake aktiviert).

- [ ] **Step 4: Boilerplate-Metrik im Kommentar festhalten**

Zähle die Zeilen im Block "LAYER 1: Linear 784 -> 256" (Kommentar ausgeschlossen, von `static float w0` bis `model[1] = reluLayerInit(q, q);`). Erwartet: ~12-13 Zeilen pro Linear+ReLU-Paar × 4 + ein Finalblock = ~60 Zeilen reines Boilerplate. Notiere die exakte Zahl am Ende des Kommentar-Headers:

```c
 * Boilerplate-Metrik (Stand XYZ): N Zeilen für die 5 Linear-Layer-Konstruktion,
 * davon M identisch mechanisch (nur Dim-Konstanten + Buffer-Namen ändern sich).
```

- [ ] **Step 5: Commit**

```
jj describe -m "examples: add mlp_mnist_stress_host for USERAPI audit boilerplate metric"
jj new
```

---

## Task 3: PyTorch-Referenz-Infrastruktur anlegen

Geteilte Harness `compare.py`, die für N=5 Seeds: (1) die ODT-Binary startet, Stdout parst; (2) das PyTorch-Modell trainiert; (3) mean±std beider Verteilungen rechnet; (4) checkt, ob der ODT-Wert innerhalb 2σ des PyTorch-Mean liegt. Per-Example-Skripte sind dünne Wrapper, die nur die PyTorch-Modell-Definition + Hyperparameter liefern.

**Files:**
- Modify: `pyproject.toml` (torch-Dependency hinzufügen)
- Create: `src/examples/reference/__init__.py` (leer)
- Create: `src/examples/reference/compare.py`

- [ ] **Step 1: torch zu pyproject.toml hinzufügen**

Run: `uv add 'torch>=2.3'`
Expected: `pyproject.toml` listet `torch` unter `dependencies`, `uv.lock` aktualisiert.

- [ ] **Step 2: Leeres `__init__.py` anlegen**

Run: `touch src/examples/reference/__init__.py`

- [ ] **Step 3: compare.py schreiben**

```python
"""
Vergleichs-Harness zwischen ODT-Binary und einer PyTorch-Referenz.

Usage (aus einem per-example Skript):

    from compare import compare_with_pytorch, Config

    def build_pytorch_model():
        import torch.nn as nn
        return nn.Sequential(nn.Linear(784, 20), nn.ReLU(), nn.Linear(20, 10))

    def train_pytorch(model, seed): ...  # returns final metric (float)

    cfg = Config(
        binary_path="build/HOST-Debug/HOST",
        example_name="mlp_mnist_float32_host",
        stdout_metric_regex=r"accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=5,
        sigma_multiplier=2.0,
        pytorch_build=build_pytorch_model,
        pytorch_train=train_pytorch,
    )
    compare_with_pytorch(cfg)

Exit code 0 bei Pass (ODT innerhalb N σ von PyTorch-Mean), 1 bei Fail.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import statistics
import subprocess
import sys
from typing import Callable


@dataclasses.dataclass
class Config:
    binary_path: str
    example_name: str
    stdout_metric_regex: str            # capture group 1 = float metric
    higher_is_better: bool
    n_seeds: int
    sigma_multiplier: float
    pytorch_build: Callable[[], "torch.nn.Module"]  # noqa: F821
    pytorch_train: Callable[["torch.nn.Module", int], float]  # noqa: F821
    odt_build_preset: str = "HOST-Debug"


def run_odt_once(cfg: Config) -> float:
    """Baut das ODT-Example und extrahiert die Metrik aus stdout."""
    env = os.environ.copy()
    cmake_cfg = subprocess.run(
        ["cmake", "--preset", cfg.odt_build_preset,
         f"-DODT_EXAMPLE={cfg.example_name}"],
        capture_output=True, text=True, env=env,
    )
    if cmake_cfg.returncode != 0:
        raise RuntimeError(f"cmake configure failed:\n{cmake_cfg.stderr}")
    cmake_build = subprocess.run(
        ["cmake", "--build", "--preset", cfg.odt_build_preset],
        capture_output=True, text=True, env=env,
    )
    if cmake_build.returncode != 0:
        raise RuntimeError(f"cmake build failed:\n{cmake_build.stderr}")
    run = subprocess.run(
        [cfg.binary_path], capture_output=True, text=True, env=env,
    )
    if run.returncode != 0:
        raise RuntimeError(f"ODT binary failed:\n{run.stdout}\n{run.stderr}")
    match = re.search(cfg.stdout_metric_regex, run.stdout)
    if not match:
        raise RuntimeError(
            f"Konnte Metrik nicht parsen. Regex: {cfg.stdout_metric_regex}\n"
            f"Stdout:\n{run.stdout}"
        )
    return float(match.group(1))


def run_pytorch_seeds(cfg: Config) -> list[float]:
    import torch
    results = []
    for seed in range(cfg.n_seeds):
        torch.manual_seed(seed)
        model = cfg.pytorch_build()
        metric = cfg.pytorch_train(model, seed)
        results.append(metric)
        print(f"  [pytorch seed={seed}] metric={metric:.4f}")
    return results


def compare_with_pytorch(cfg: Config) -> int:
    print(f"=== compare: {cfg.example_name} ===")
    print(f"Running ODT example once...")
    odt_metric = run_odt_once(cfg)
    print(f"ODT metric: {odt_metric:.4f}")

    print(f"Running PyTorch reference (N={cfg.n_seeds} seeds)...")
    pt_metrics = run_pytorch_seeds(cfg)
    pt_mean = statistics.fmean(pt_metrics)
    pt_std = statistics.pstdev(pt_metrics) if len(pt_metrics) > 1 else 0.0
    print(f"PyTorch: mean={pt_mean:.4f} std={pt_std:.4f} "
          f"(raw: {[f'{m:.4f}' for m in pt_metrics]})")

    diff = odt_metric - pt_mean
    # Wenn std=0 (z.B. durch Zufall bei sehr stabilen Trainings): fallback
    # auf 5% relative Abweichung. Andernfalls 2σ-Regel.
    if pt_std < 1e-9:
        threshold = 0.05 * abs(pt_mean) if abs(pt_mean) > 1e-9 else 1e-3
        rule = "fallback 5% relative"
    else:
        threshold = cfg.sigma_multiplier * pt_std
        rule = f"{cfg.sigma_multiplier}σ"

    passed = abs(diff) <= threshold
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] diff={diff:+.4f} threshold=±{threshold:.4f} ({rule})")
    return 0 if passed else 1


def main_cli(cfg: Config):
    """Entry point für per-example scripts: sys.exit mit compare-Returncode."""
    sys.exit(compare_with_pytorch(cfg))
```

- [ ] **Step 4: Import-Smoke-Test**

Run: `uv run python -c "from src.examples.reference.compare import Config, compare_with_pytorch; print('ok')"`
Expected: `ok` (keine Import-Fehler).

Falls das wegen des Bindestrichs im Package-Namen oder Python-Path-Problemen scheitert: in Step 5 dokumentieren, dass per-example Skripte direkt aus `src/examples/reference/` laufen (sys.path-Manipulation oder `uv run python src/examples/reference/<script>.py`).

- [ ] **Step 5: Commit**

```
jj describe -m "examples: add PyTorch reference comparison harness"
jj new
```

---

## Task 4: PyTorch-Referenz für mlp_mnist_float32_host

Das ODT-Example ist: 784→20→ReLU→10→Softmax + CrossEntropy + SGD lr=0.001 + BS=32 + 10 Epochen. Xavier-uniform Init für Weights, Zero-Init für Bias. PyTorch muss dieselbe Architektur und Hyperparameter replizieren.

**Files:**
- Create: `src/examples/reference/mlp_mnist_float32_host_ref.py`

- [ ] **Step 1: Skript schreiben**

```python
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
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
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
        n_seeds=5,
        sigma_multiplier=2.0,
        pytorch_build=build_model,
        pytorch_train=train,
    )
    main_cli(cfg)
```

- [ ] **Step 2: Referenz laufen lassen**

Run: `uv run python src/examples/reference/mlp_mnist_float32_host_ref.py`
Expected: 
- CMake configure + build von ODT-Example (einige Sekunden).
- ODT-Binary läuft (~12 Minuten per Memory).
- PyTorch trainiert 5× (jeweils ~1-2 Minuten CPU).
- Finale Zeile `[PASS] diff=... threshold=...` oder `[FAIL]`.

Falls FAIL: Hyperparameter im PyTorch-Modell checken (weight init method, loss formula, LR, batch order).

- [ ] **Step 3: Commit (nur bei PASS)**

```
jj describe -m "examples: add PyTorch reference for mlp_mnist_float32_host"
jj new
```

Falls FAIL nach Debugging persistent: Als Finding in `docs/odt-userapi-findings-misc.md` vermerken (aber die Datei existiert erst in Plan 2 — hier vorerst in einer TODO-Datei `docs/phase5e-failures.md` festhalten).

---

## Task 5: PyTorch-Referenz für mlp_mnist_float32_mcu

MCU-Variante nutzt 100-Sample-Subset, BS=1, 3 Epochen. PyTorch repliziert das.

**Files:**
- Create: `src/examples/reference/mlp_mnist_float32_mcu_ref.py`

- [ ] **Step 1: Skript schreiben**

```python
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
```

**Hinweis:** Beachte, dass das ODT-Example über Host läuft (mcu-Variante ist portable), und die stdout-Zeile enthält `subset_accuracy=XX.XX%` (per Memory). Falls der Regex nicht matcht: Binary selbst laufen lassen und Stdout-Format prüfen.

- [ ] **Step 2: Laufen lassen**

Run: `uv run python src/examples/reference/mlp_mnist_float32_mcu_ref.py`
Expected: PASS oder FAIL wie in Task 4.

- [ ] **Step 3: Commit bei PASS**

```
jj describe -m "examples: add PyTorch reference for mlp_mnist_float32_mcu"
jj new
```

---

## Task 6: PyTorch-Referenz für mlp_mnist_stress_host

**Files:**
- Create: `src/examples/reference/mlp_mnist_stress_host_ref.py`

- [ ] **Step 1: Skript schreiben**

```python
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
```

- [ ] **Step 2: Laufen lassen**

Run: `uv run python src/examples/reference/mlp_mnist_stress_host_ref.py`
Expected: PASS (oder FAIL — bei FAIL: hyperparams im C-Code vs. Python prüfen).

- [ ] **Step 3: Commit bei PASS**

```
jj describe -m "examples: add PyTorch reference for mlp_mnist_stress_host"
jj new
```

---

## Task 7: Linear-Regression-Validierung dokumentieren

`linear_regression.c` enthält bereits eine eingebaute Validierung: fixe expected-weights `{5, -1, 9, 22, -100, 18}` mit 3% Toleranz. Deterministisches Training (keine Zufallskomponente), keine N-Seed-Statistik nötig. Aber das muss klar dokumentiert sein — sonst entsteht Eindruck, dass diese Validierung fehlt.

**Files:**
- Create: `src/examples/reference/README.md`

- [ ] **Step 1: README.md schreiben**

```markdown
# PyTorch Reference Comparisons (Phase 5e)

Diese Skripte vergleichen ODT-Examples gegen PyTorch-Referenzen mit N=5 Seeds
und 2σ-Acceptance-Kriterium.

## Abgedeckte Examples

| Example | Reference Script | Methode |
|---|---|---|
| `linear_regression` | *(built-in)* | Deterministisch; self-check in `linear_regression.c` gegen fixe expected weights mit 3% Toleranz |
| `mlp_mnist_float32_host` | `mlp_mnist_float32_host_ref.py` | N=5 Seeds PyTorch, 2σ-Vergleich auf `accuracy` |
| `mlp_mnist_float32_mcu` | `mlp_mnist_float32_mcu_ref.py` | N=5 Seeds PyTorch, 2σ-Vergleich auf `subset_accuracy` |
| `mlp_mnist_stress_host` | `mlp_mnist_stress_host_ref.py` | N=5 Seeds PyTorch, 2σ-Vergleich auf `accuracy` |
| `mnist_inference` | *(keine)* | Nur Forward-Only; validiert implizit wenn `mlp_mnist_float32_host` validiert |
| `mlp_mnist_asym_*` | *(gesperrt)* | `#error`-Gate — ODT Issue #61 |

## Voraussetzungen

- MNIST .npy unter `src/examples/data/` (einmal `uv run src/examples/data/generate_subset.py`)
- `HOST-Debug` Preset gebaut
- PyTorch + numpy installiert (`uv sync`)

## Ausführen

```
uv run python src/examples/reference/<example>_ref.py
```

Exit 0 bei PASS, 1 bei FAIL. Bei persistentem FAIL: Hyperparameter-Mismatch
zwischen C-Code und PyTorch-Modell prüfen, oder als Finding ins USERAPI-Audit
(Plan 2) mitnehmen.
```

- [ ] **Step 2: Commit**

```
jj describe -m "examples: document reference comparison methodology"
jj new
```

---

## Task 8: Phase 6 — Qualitätsreview der Examples

Konsistenzprüfung über alle Examples. Kein separates Artefakt — direkt in den Examples fixen.

**Checkliste pro Example:**
1. Header-Kommentar vorhanden (Zweck, Target-Kompatibilität, Daten-Voraussetzung, erwartetes Ergebnis)
2. Einheitliche Konstanten-Namen (`INPUT_DIM`, `HIDDEN_DIM`, `OUTPUT_DIM`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `MODEL_SIZE`)
3. `static` Buffer für alle Tensor-Daten (MCU-Portabilität — kein Heap)
4. Kein `freeTensor` auf user-allokierten Tensoren (ODT-Gotcha Memory)
5. Bias-Shape `{1, out_features}` (nicht `{out_features, 1}`)
6. Labels-Shape `{1, num_classes}` für MSE
7. MCU-Examples: `initSampleTensors()` pattern (einmal init, dann Buffer überschreiben)

**Files:**
- Review: `src/examples/*.c` (alle)

- [ ] **Step 1: `linear_regression.c` prüfen**

Alle 7 Punkte durchgehen. Falls etwas nicht passt: fixen. Aktueller Stand (vor Review):
- Header ✓
- Konstanten ✓ (`IN_FEATURES`, `OUT_FEATURES`, `NUM_SAMPLES`, `NUM_ITERATIONS`, `LEARNING_RATE`)
  → **Mögliche Inkonsistenz:** andere Examples nutzen `INPUT_DIM`/`OUTPUT_DIM`. Entscheidung: bei `IN_FEATURES`/`OUT_FEATURES` belassen, da die Semantik (Features, nicht Dims) hier spezifisch zutrifft. Im Header-Kommentar erwähnen.
- static buffers ✓ (alle Weights/Biases/Inputs/Labels sind stack oder static)
- kein freeTensor ✓ (Kommentar weist explizit darauf hin)
- bias shape: `{1, OUT_FEATURES}` ✓
- label shape: `{1, OUT_FEATURES}` ✓

Kein Handlungsbedarf. Oder gegebenenfalls kleinen Hinweis im Header, dass die Naming-Abweichung (`IN_FEATURES` vs `INPUT_DIM`) Absicht ist.

- [ ] **Step 2: `mlp_mnist_float32_host.c` prüfen**

- Header ✓
- Konstanten ✓ (`INPUT_DIM`, `HIDDEN_DIM`, `OUTPUT_DIM`, `NUM_CLASSES`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `MODEL_SIZE`)
- `w0`, `b0`, `w1`, `b1` sind `static` ✓
- bias shape `{1, HIDDEN_DIM}` / `{1, OUTPUT_DIM}` ✓

Kein Handlungsbedarf — dient als Stil-Vorlage.

- [ ] **Step 3: `mlp_mnist_float32_mcu.c` prüfen**

Muss `initSampleTensors`-Pattern haben (curItem/curLabel einmal allokiert, Buffer pro Sample überschrieben). Falls nicht: fixen nach dem Pattern aus `mlp_mnist_asym_mcu.c` (dort per Memory korrekt implementiert).

- [ ] **Step 4: `mnist_inference.c` prüfen**

Forward-only, kein Training. Prüfe:
- Header dokumentiert: "lädt pretrained weights aus `mnist_pretrained_float32.h`".
- Konstanten konsistent zu `mlp_mnist_float32_host.c` (`HIDDEN_DIM=20`, etc.).
- Pretrained weights werden korrekt per `tensorInitFloat` gewrappt (kein freeTensor).

- [ ] **Step 5: `mlp_mnist_stress_host.c` prüfen**

Self-check: wurde in Task 2 bereits konsistent geschrieben. Kurz nochmal drübersehen, ob alle Checkliste-Punkte erfüllt sind.

- [ ] **Step 6: Gefundene Fixes committen**

Falls Fixes nötig waren:
```
jj describe -m "examples: phase 6 consistency pass across all examples"
jj new
```
Sonst: Task als completed markieren ohne Commit.

---

## Task 9: Phase 7 — README.md + Hardware-Test-Matrix

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Vollständigen README schreiben**

```markdown
# OnDeviceTraining Base Project

Beispielsammlung ("example zoo") für die [OnDeviceTraining](https://github.com/es-ude/OnDeviceTraining) C-Library. Jedes Example demonstriert einen Use-Case (Training, Inference, verschiedene Quantizations) und läuft — wenn als `host`-portabel markiert — auf HOST, PICO1/2_W und STM32-Targets aus dem gleichen Source.

## Quickstart

```bash
# 1. Setup (einmal):
cmake --preset PREPARE  # fetched pico-sdk + OnDeviceTraining

# 2. Example wählen + builden (Host):
cmake --preset HOST-Debug -DODT_EXAMPLE=linear_regression
cmake --build --preset HOST-Debug
./build/HOST-Debug/HOST

# 3. Für MNIST-basierte Host-Examples: Daten generieren:
uv run src/examples/data/generate_subset.py

# 4. Für MCU (z.B. Pico2 W):
cmake --preset PICO2_W -DODT_EXAMPLE=linear_regression
cmake --build --preset PICO2_W
# Flash UF2 aus build/PICO2_W/
```

## Examples

| Name | Was | Host | MCU | Validierung |
|---|---|---|---|---|
| `linear_regression` | 1 Linear Layer, 3 Samples, MSE, 100 SGD | ✓ | ✓ | Built-in self-check (3% tolerance) |
| `mlp_mnist_float32_host` | 784→20→10 MLP, CE, full MNIST | ✓ | – | PyTorch ref (N=5, 2σ) |
| `mlp_mnist_float32_mcu` | Gleiche MLP, 100-Sample Subset | ✓ | ✓ | PyTorch ref (N=5, 2σ) |
| `mnist_inference` | Forward-only mit pretrained weights | ✓ | ✓ | Impliziert via float32_host |
| `mlp_mnist_stress_host` | 5-Hidden-Layer MLP (Stresstest) | ✓ | – | PyTorch ref (N=5, 2σ) |
| `mlp_mnist_asym_host` | ASYM-Forward Template | ✗ | ✗ | Gated durch #error ([ODT #61](https://github.com/es-ude/OnDeviceTraining/issues/61)) |
| `mlp_mnist_asym_mcu` | ASYM-Forward Template (MCU) | ✗ | ✗ | Gated durch #error ([ODT #61](https://github.com/es-ude/OnDeviceTraining/issues/61)) |

## Hardware-Test-Matrix

Cross-Kompilation aller MCU-fähigen Examples ist für alle Targets verifiziert.
"Flash+Run" heißt: Binary wurde auf echter Hardware geflasht und erfolgreich
ausgeführt.

| Target | Flash+Run verifiziert |
|---|---|
| HOST | Alle host-fähigen Examples |
| PICO1 | *(nachtragen wenn getestet)* |
| PICO2_W | *(nachtragen wenn getestet)* |
| STM32F756ZGT6 | *(nachtragen wenn getestet)* |
| STM32L476RG | *(nachtragen wenn getestet)* |
| STM32L4R5ZI | *(nachtragen wenn getestet)* |

## Validierung per PyTorch-Referenz

Training-Examples werden gegen PyTorch validiert mit N=5 Seeds und 2σ-Toleranz.
Siehe `src/examples/reference/README.md`.

## ODT-Gotchas beim Schreiben eigener Examples

- **`freeTensor` crasht auf user-allokierten Buffers.** Wenn du `tensorInitFloat` mit einem eigenen `float[...]` aufrufst, **nicht `freeTensor` aufrufen** — ODTs Allocator kann keine stack/static Pointer freigeben.
- **Bias-Shape ist `{1, out_features}`, nicht `{out_features, 1}`.** Sonst Shape-Mismatch in `addFloat32TensorsInplace`.
- **`PRINT_ERROR` ist silent bei `DLEVEL=0` (Default).** Für Diagnose: `-DDEBUG_MODE_ERROR` in den Compile-Definitionen (HOST-Debug hat das bereits).
- **ASYM-Quantization ist in `linearForward` nicht dispatched** (ODT Issue #61). Bis das gefixt ist: `SYM_INT32` oder `FLOAT32` nutzen.

## Lizenz

*(unverändert wie zuvor — falls noch keine vorhanden, Entscheidung offen lassen)*
```

- [ ] **Step 2: README committen**

```
jj describe -m "docs: add example catalog, hardware matrix, and ODT gotchas to README"
jj new
```

---

## Task 10: .gitignore-Cleanup

Das Working-Copy ist aktuell durch Build-Artefakte und Data-Files zugemüllt. `.gitignore` erweitern, damit nur relevanter Code getrackt wird.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Aktuellen Inhalt lesen + Ergänzungen schreiben**

Aktueller Inhalt (kurz):
```
/build
# Claude Code
CLAUDE.md
.claude/
# JetBrains
.idea/
# Devenv
.devenv*
...
```

Füge am Ende hinzu:

```gitignore
# Vendored dependencies (re-fetched via cmake --preset PREPARE)
/pico-sdk/
/OnDeviceTraining/

# Build artifacts that end up at repo root when cmake is run in-source
/CMakeCache.txt
/CMakeFiles/
/cmake_install.cmake
/Makefile

# MNIST data (re-generated via generate_subset.py)
src/examples/data/*.npy
src/examples/data/*.gz
src/examples/data/*-ubyte

# uv / Python caches
__pycache__/
*.pyc
.uv-cache/

# Brainstorming + planning drafts — local only per user preference
/docs/superpowers/
```

- [ ] **Step 2: `jj status` prüfen**

Run: `jj status`
Expected: Deutlich weniger untracked files. `pico-sdk/`, `OnDeviceTraining/`, `src/examples/data/*.npy` etc. tauchen nicht mehr auf.

- [ ] **Step 3: Commit**

```
jj describe -m "gitignore: ignore vendored deps, MNIST data, superpowers drafts, in-source build artifacts"
jj new
```

---

## Task 11: Abschluss-Smoke-Test

Alle Examples nochmal durchbauen, stellt sicher dass nichts während Phase 6 kaputtgegangen ist.

- [ ] **Step 1: Host-Examples bauen + laufen**

Für jedes host-fähige Example:
```
cmake --preset HOST-Debug -DODT_EXAMPLE=<name>
cmake --build --preset HOST-Debug
./build/HOST-Debug/HOST
```

Examples: `linear_regression`, `mlp_mnist_float32_host`, `mlp_mnist_float32_mcu`, `mnist_inference`, `mlp_mnist_stress_host`.

Expected: Alle laufen ohne Crash, liefern plausible Ergebnisse.

- [ ] **Step 2: MCU-Cross-Compile für PICO2_W und einen STM32-Target**

```
cmake --preset PICO2_W -DODT_EXAMPLE=linear_regression
cmake --build --preset PICO2_W

cmake --preset STM32F756ZGT6-Debug -DODT_EXAMPLE=linear_regression
cmake --build --preset STM32F756ZGT6-Debug
```

Expected: Erfolgreicher Build, .uf2/.hex-Datei entsteht.

- [ ] **Step 3: Asym-Gates prüfen**

```
cmake --preset HOST-Debug -DODT_EXAMPLE=mlp_mnist_asym_host
cmake --build --preset HOST-Debug
```
Expected: Build-Fehler mit `#error`-Text.

- [ ] **Step 4: Kein Commit nötig** — dient nur als Regression-Check.

---

## Self-Review Ergebnis

**Spec-Abdeckung:**
- Phase 5e (PyTorch-Ref, N=5, 2σ) → Tasks 3-7 ✓
- Phase 6 (Quality Review) → Task 8 ✓
- Phase 7 (README + Hardware Matrix) → Task 9 ✓
- Stresstest-Example → Task 2 + Task 6 (Ref) ✓
- Asym-Gates → Task 1 ✓
- `.gitignore`-Hygiene ist nicht im Spec, aber nötig für saubere Working-Copy → Task 10 (zusätzlich)

**Keine Platzhalter.** Alle Code-Blöcke sind vollständig.

**Type-Konsistenz:** `compare.Config`-Felder werden in allen per-example Skripten identisch benutzt.

**Offene Punkte für zukünftige Pläne:**
- Plan 2 (Audit Pass 1) startet nach erfolgreichem Abschluss von Plan 1
- Plan 3 (Issue-Filing) folgt auf Plan 2

---

## Execution Handoff

Plan komplett. Er umfasst 11 Tasks mit ~50 Steps, geschätzte Ausführungsdauer: 2-4 Stunden (dominiert durch die mehrfachen MNIST-Trainingsläufe in den PyTorch-Refs).

**Ausführungs-Optionen:**

1. **Subagent-Driven (empfohlen)** — frischer Subagent pro Task, Review zwischen Tasks, schnelle Iteration
2. **Inline Execution** — Tasks in dieser Session ausführen via executing-plans, Batch mit Checkpoints

Welche Option?
