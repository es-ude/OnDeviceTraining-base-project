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
