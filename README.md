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
