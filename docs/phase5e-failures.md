# Phase 5e Comparison Failures

Zeitbegrenzter Ablage-Ort für ODT-vs-PyTorch-Vergleiche, die das 2σ-Akzeptanzkriterium nicht erfüllen. Diese Findings wandern in Plan 2 (USERAPI-Audit) in die endgültige Issue-Dokumentation.

## 2026-04-19 — `mlp_mnist_float32_host`

**Setup:** 784→20→ReLU→10→Softmax + CrossEntropy + SGD lr=0.001 + BS=32 + 10 Epochen, Xavier-uniform / Zero-Bias Init, voller MNIST-Trainset.

**Ergebnis:**
- ODT accuracy: **86.00%**
- PyTorch mean (N=5 Seeds): **89.31% ± 0.31%**
- diff: **-3.31 Prozentpunkte** (ODT schlechter)
- 2σ-Toleranz: **±0.62 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 10.7σ außerhalb)

**PyTorch-Rohwerte:** `[89.77, 89.03, 89.38, 89.46, 88.91]` — sehr geringe Streuung, klarer systematischer Unterschied.

**Hypothesen (unverifiziert):**
1. Unterschied in der Cross-Entropy-Gradientenberechnung gegenüber PyTorchs `nll_loss(log(softmax(.)))`.
2. Unterschied in der Xavier-Uniform-Init-Formel (fan_in vs. fan_in+fan_out Skalierung).
3. Unterschied in der Shuffle-Semantik des DataLoaders (evtl. unterschiedliche Batch-Zusammenstellungen über Epochen).
4. Akkumulations-/Rounding-Unterschiede im CE-Loss auf großen Batches.

**Nächste Schritte (Plan 2):** Isoliertes Repro-Micro-Example bauen, das nach 1 Epoche den State (Weights, Gradienten, Loss) exakt abgreift und gegen PyTorch numerisch vergleicht, um die Divergenz-Quelle einzugrenzen.

**Commit mit Script:** siehe `src/examples/reference/mlp_mnist_float32_host_ref.py` (wurde trotz FAIL committed — das Script selbst ist korrekt, FAIL ist ein echtes ODT-Finding).

## 2026-04-19 — `mlp_mnist_float32_mcu`

**Setup:** 784→20→ReLU→10→Softmax + CrossEntropy + SGD lr=0.01 + BS=1 + 3 Epochen, Xavier-uniform / Zero-Bias Init, 100-Sample-Trainset / 20-Sample-Testset (MCU-Budget-Subset).

**Ergebnis:**
- ODT accuracy: **30.00%**
- PyTorch mean (N=5 Seeds): **63.00% ± 8.12%**
- diff: **-33.00 Prozentpunkte** (ODT schlechter)
- 2σ-Toleranz: **±16.25 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 4.06σ außerhalb)

**PyTorch-Rohwerte:** `[65.00, 50.00, 65.00, 60.00, 75.00]` — hohe Streuung erwartbar bei 20 Test-Samples, aber ODT liegt deutlich unterhalb der gesamten PyTorch-Verteilung (min=50.00%).

**Zweites Signal für denselben Befund:** Task 4 (volles MNIST, BS=32, lr=0.001) zeigte -3.31 Prozentpunkte. Task 5 (100-Sample-Subset, BS=1, lr=0.01) zeigt -33.00 Prozentpunkte. Zwei unterschiedliche Konfigurationen, beide ODT schlechter — das spricht für einen systematischen Unterschied, nicht für Hyperparameter-Sensitivität.

**Hypothesen (unverifiziert, zusätzlich zu Task 4):**
1. BS=1 verstärkt Unterschiede in der CE-Gradientenberechnung (pro-Sample Gradient statt Batch-Mittel).
2. lr=0.01 × BS=1 mit nur 100 Samples ist numerisch empfindlich — Einzel-Step-Fehler akkumulieren.
3. Weights-Initialisierung könnte bei kleinen Modellen (784→20→10) kritischer sein als bei größeren.

**Nächste Schritte (Plan 2):** Gemeinsam mit Task-4-Finding in das USERAPI-Audit-Repro-Beispiel. Beide Szenarien sollten durch dasselbe Micro-Example (1 Epoche, State-Dump, Numerik-Vergleich) reproduzierbar sein.

**Commit mit Script:** siehe `src/examples/reference/mlp_mnist_float32_mcu_ref.py` (FAIL-Marker im Commit).

## 2026-04-19 — `mlp_mnist_stress_host`

**Setup:** 784→256→128→64→32→10 + 4× ReLU + Softmax + CrossEntropy + SGD lr=0.001 + BS=32 + 5 Epochen, Xavier-uniform / Zero-Bias Init, voller MNIST-Trainset (~244K Parameter).

**Ergebnis:**
- ODT accuracy: **78.04%**
- PyTorch mean (N=5 Seeds): **89.87% ± 0.36%**
- diff: **-11.83 Prozentpunkte** (ODT schlechter)
- 2σ-Toleranz: **±0.72 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 33σ außerhalb)

**PyTorch-Rohwerte:** `[90.51, 89.88, 89.51, 89.55, 89.92]` — sehr geringe Streuung, klarer systematischer Unterschied.

**Drittes Signal — Pattern verdichtet sich:**

| Example | Tiefe | Absolute Lücke | σ-Abstand |
|---|---|---|---|
| `mlp_mnist_float32_host` | 2-Layer (784→20→10) | -3.31pp | 10.7σ |
| `mlp_mnist_float32_mcu` | 2-Layer (Subset BS=1) | -33.00pp | 4.06σ |
| `mlp_mnist_stress_host` | 5-Layer (784→256→128→64→32→10) | -11.83pp | 33σ |

Die 5-Layer-Lücke ist 3.6× größer als die 2-Layer-Lücke bei identischen Hyperparametern (BS=32, lr=0.001, voller MNIST). Das spricht dafür, dass der ODT-PyTorch-Unterschied pro Layer akkumuliert — konsistent mit einer Gradientenberechnungs-Divergenz, die sich über Backprop-Ketten multipliziert.

**Hypothesen (verdichtet aus Task 4 + 5 + 6):**
1. **Gradientenakkumulation über Layer**: CE-Gradient weicht ab, Unterschied wächst pro Layer durch Backprop.
2. **Xavier-Uniform-Formel**: Möglicherweise unterschiedliche fan_in/fan_out-Skalierung, mit größerem Effekt bei tieferen Netzen.
3. **DataLoader-Shuffle-Semantik**: Nur bei unterschiedlichen Batch-Kompositionen über Epochen hinweg, nicht pro Layer — unwahrscheinlicher als Haupt-Ursache.
4. **Float32-Akkumulations-Rundung in MatMul**: Bei 256×128 MatMul größer als bei 20×10 — erklärt den Tiefen-Effekt.

**Nächste Schritte (Plan 2 USERAPI-Audit):** Das Repro-Micro-Example **muss** layer-depth-sweep enthalten (1, 2, 5 Layer, gleiche Hyperparams), um zu verifizieren dass die Lücke tatsächlich pro Layer wächst. Hypothese 1 oder 4 hat dann einen quantitativen Fingerabdruck.

**Commit mit Script:** siehe `src/examples/reference/mlp_mnist_stress_host_ref.py` (FAIL-Marker im Commit).
