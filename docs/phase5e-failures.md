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
