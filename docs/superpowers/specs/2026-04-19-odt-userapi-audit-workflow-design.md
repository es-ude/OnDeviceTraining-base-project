# ODT USERAPI Audit Workflow — Design

**Status:** Approved (2026-04-19)
**Author:** Leo Buron (mit Claude)
**Repo:** `OnDeviceTraining-base-project` (dieses Repo)
**Zielrepo für Issues:** `github.com/es-ude/OnDeviceTraining`

## Kontext und Ziel

Das `OnDeviceTraining-base-project` Repo dient zwei Zwecken:

1. **Primär:** Beispielsammlung ("example zoo") für die OnDeviceTraining-Library, die zeigt, wie man MCU-taugliche neuronale Netze mit ODT baut — auf HOST validiert und auf mehreren MCU-Targets cross-kompiliert.
2. **Sekundär (und treibend für dieses Dokument):** Audit-Vehikel für die USERAPI der ODT-Library. Beim Schreiben der Beispiele wird offensichtlich, wo die USERAPI den User zwingt, viel mechanischen C-Code zu schreiben, der später schwer debugbar ist. Diese Befunde sollen systematisch aufgenommen und als GitHub-Issues im ODT-Repo eingebracht werden, wo eine andere Claude-Instanz sie abarbeitet.

**Primäres Audit-Ziel:** Identifikation **fehlender Abstraktionen** in der USERAPI, deren Abwesenheit zu großen, schwer debugbaren C-Code-Mengen im User-Code führt.

**Sekundäres Audit-Ziel:** Sammlung von Bugs, silent failures, API-Inkonsistenzen, die beim Tracen der Dispatch-Pfade nebenbei auffallen.

Bereits in einer früheren Session gefundene Issues (ohne systematisches Audit):
- `linearForward` dispatched ASYM nicht → silent `exit(1)` (getrackt als ODT #61)
- `linearForwardSymInt32` dereferenziert `qConfig` unbedingt → Segfault bei float-initialisierten Weights
- `tensorInitWithDistribution` nimmt Quantization, propagiert sie aber nicht an Dispatch
- `freeTensor` crasht auf user-allokierten Buffern
- `PRINT_ERROR` silent bei `DLEVEL=0` (Default)

Diese bestätigen, dass ein systematisches Audit Wert produziert.

## 1. Workflow-Überblick

**Zwei Tracks, sequenziell (Approach A):**

```
Track 1 (Examples)  →  Track 2 (USERAPI Audit)  →  Issue Filing
  Phase 5e (Validierung)    Pass 1 Methodik          Batch-Review
  Phase 6 (Qualität)        Stresstest liefert      Human-Gate
  Phase 7 (Zusammenfassung) Boilerplate-Metriken    gh issue create
```

Begründung für sequenziell: Die Examples sind die Basis-Artefakte, auf die sich das Audit stützt. Der Stresstest insbesondere ist das Hauptmessinstrument des Audits. Ohne stabile Examples hat das Audit keinen Boden.

**Human-in-the-Loop Gates:**
- Nach Phase 5e: Validierungsergebnisse gegen PyTorch-Referenz bestätigen
- Nach Audit Pass 1: User liest `odt-userapi-abstractions.md` + `findings-misc.md` durch
- Vor `gh issue create`: User reviewt jedes proposed Issue einzeln

## 2. Phase 5e — Validierung der Examples

**Ziel:** Sicherstellen, dass die Examples nicht nur kompilieren, sondern korrekte Ergebnisse produzieren — robust gegen stochastische Varianz.

**Vorgehen: PyTorch-Referenz-Regression**

Pro Example, das Training beinhaltet (`linear_regression`, `mlp_mnist_float32_host`, `mlp_mnist_float32_mcu`):

1. PyTorch-Skript unter `src/examples/reference/` schreibt äquivalentes Modell mit gleicher Architektur, Init, Optimizer, LR.
2. **N=5 Seeds** laufen für PyTorch und ODT.
3. Aggregiert wird: mean ± std der Final-Metrik (Loss oder Accuracy).
4. **Akzeptanzkriterium:** ODT-Ergebnis muss innerhalb **2σ** der PyTorch-Verteilung liegen.
5. Falls divergent: N erhöhen (→ 10, 20) um sicherzustellen, dass es keine stochastische Anomalie ist. Reale Divergenz wird als Issue im ODT-Repo gefiled.

**Erwartung:** ODT und PyTorch sollten nicht bitgleich sein (unterschiedliche RNG, Reduktions-Reihenfolgen), aber statistisch überlappen.

**Python-Umgebung:** `uv` (siehe globale Preferences), keine `pip`/`venv`.

## 3. Phase 6 — Qualitätsreview der Examples

**Ziel:** Konsistenz und Lesbarkeit über alle Examples hinweg. Nicht Code-Korrektheit (das hat Phase 5e), sondern Stil.

**Checkliste pro Example:**
- Einheitlicher Header-Kommentar (Zweck, Target, Daten-Quelle, erwartetes Ergebnis)
- Konsistente Benennung der Konstanten (`INPUT_DIM`, `HIDDEN_DIM`, etc.)
- `static` Buffer für alle Tensor-Backing-Stores (MCU-Portierbarkeit)
- Kein `freeTensor` auf user-allokierten Tensoren (ODT-Gotcha #1)
- Bias/Label-Shape: `{1, N}` nicht `{N, 1}` (ODT-Gotcha #2)
- MCU-Examples: Sample-Tensor einmal in `initSampleTensors()`, danach nur Buffer überschreiben
- Cross-Kompilation für alle Targets grün, auch wenn Hardware nicht verfügbar

**Output:** Kein separates Artefakt — direkt in Code korrigiert und committet.

## 4. Phase 7 — Zusammenfassung der Examples

**Ziel:** Projekt-externe Benutzbarkeit herstellen.

**Arbeiten:**
- `README.md` ausbauen: Example-Tabelle mit Zweck, Target-Matrix, erwarteter Laufzeit, erwartetem Ergebnis
- **Hardware-Test-Matrix:** explizit dokumentieren, welche Examples auf welchem Target tatsächlich geflasht + gelaufen sind vs. nur cross-kompiliert
- `CMakeLists.txt` aufräumen falls nötig
- Finale Commits, alle Änderungen in jj konsistent

**Explizit nicht im Scope:**
- Publikation oder Release-Tag
- CI-Setup (das gehört in eine separate Session)

## 5. Stresstest-Example (`mlp_mnist_stress_host.c`)

**Zweck:** Konkrete Messlatte für Boilerplate-Dichte. Ohne dieses Example ist die Audit-Behauptung "USERAPI produziert viel undebugbaren Code" hand-wavy. Mit dem Example gibt es zählbare Zeilen-Receipts.

**Architektur:**
- 5 Hidden-Layers: 784 → 256 → 128 → 64 → 32 → 10
- ReLU zwischen allen, Softmax am Ende, CrossEntropy Loss
- FLOAT32 durchgehend (keine Quantization-Experimente, nur Boilerplate-Messung)
- MNIST aus .npy (gleiche Pipeline wie `mlp_mnist_float32_host`)

**Konstrains:**
- **Host-only.** Parameter-Count ≈ 244K × 4 Byte ≈ 970 KB → passt nicht in Pico2 W's 520 KB SRAM. `cmake/host/` HAL-Stubs reichen.
- **Keine Helper-Funktionen, keine Schleifen** für Layer-Konstruktion. Jede Layer wird explizit Zeile für Zeile aufgebaut. Das ist *gewollt* — der Boilerplate-Schmerz soll sichtbar und messbar sein, nicht von einem selbstgeschriebenen Helper versteckt.
- Direkt aus diesem Code werden im Audit die Metriken entnommen: Zeilen pro Layer, Parameter-Init-Ritual, Dim-Array-Duplikation.

**Validierung:** Teil der PyTorch-Referenz-Regression aus Phase 5e (gleiche Methodik, N=5, 2σ).

## 6. Audit Pass 1 — Methodik

**Primärlinse:** "fehlende Abstraktionen → Boilerplate → Undebugbarkeit". Alles andere ist Nebenbefund.

**Scope:** Nur die USERAPI-Header, die die 7 Beispiele tatsächlich anfassen:
- `TensorApi.h`, `QuantizationApi.h`
- `LinearApi.h`, `ReluApi.h`, `SoftmaxApi.h`
- `SgdApi.h`, `InferenceApi.h`, `TrainingLoopApi.h`, `CalculateGradsSequential.h`
- `DataLoaderApi.h`, `NPYLoaderApi.h`, `StorageApi.h`
- `LossFunction.h` (CrossEntropy, MSE)

**3-Schritt-Protokoll pro Header:**

1. **Verwendung in Examples messen** — wieviele Zeilen mechanischen Codes braucht ein typischer Use-Case? Stresstest-Example ist die Messlatte (5 Linear-Layer = maximale Boilerplate-Konzentration).
2. **Intention vs. Ritual trennen** — was davon drückt *Absicht* aus ("mach mir ein Linear 784→256 mit Xavier-Init"), was ist *Zeremonie* (static buffer, dim-array, tensorInit-with-5-args, parameterInit+gradInit-Doppelpack)?
3. **Konkrete Abstraktion vorschlagen** — welche API-Funktion würde das Ritual verschlucken? Vorher/Nachher-Zeilenzählung als Beleg. Dabei Dispatch-Pfad ab API-Call durchtracen (welche Typen werden dispatched, welche fehlen silent, wo sind Null-Checks, wo nicht).

**Primär-Artefakt: `docs/odt-userapi-abstractions.md`**

Liste von *Missing Abstractions*, jeder Eintrag:
- **Was fehlt** — z.B. "Linear-Layer-Factory, die Weight+Bias+Parameter+Grad in einem Call erzeugt"
- **Heutige Kosten** — Zeilen im Stresstest, z.B. "10 Zeilen × 5 Layer = 50 Zeilen Ritual"
- **Vorschlag** — Signatur-Skizze, z.B. `layer_t *linearLayerCreate(size_t in, size_t out, init_t, quantization_t *forwardQ, quantization_t *floatQ)`
- **Was es obviate-t** — welche Bug-Klassen verschwinden nebenbei (z.B. "keine Chance mehr auf shape-Mismatch bei Bias, weil die Factory es korrekt setzt")

**Sekundär-Artefakt: `docs/odt-userapi-findings-misc.md`**

Bugs, silent failures, lying signatures, ASYM-Dispatch etc., die beim Audit sichtbar werden, aber nicht direkt "fehlende Abstraktion" sind. Pro Eintrag:
- Reproducer-Pointer (Example + Zeile, oder ODT-Source + Zeile)
- Schweregrad (Bug / Inkonsistenz / UX)
- Relation zu existierenden ODT-Issues (#60, #61)

Grundlage für die individuellen Bug-Issues in Abschnitt 7.

**Explizit ausgenommen (Future Work → Pass 2):**
- APIs für Skip-Connections / Element-wise Add (ResNet)
- APIs für Dense-Block-Konkatenation (DenseNet)
- Convolutional Layers

Pass 1 auditiert nur, was durch konkrete Examples gestützt ist — keine spekulative Kritik an APIs, die noch nicht benutzt wurden. Pass 2 wird getriggert, wenn die ResNet/DenseNet-Examples entstehen.

## 7. Issue-Filing-Workflow

**Timing: Batch am Ende von Pass 1.** Kein drip-feed während des Audits — erst wenn beide Artefakte fertig sind, werden Issues geschrieben. Grund: manches wird beim Tracen präziser oder entpuppt sich als Duplikat eines existierenden Issues.

**Issue-Mapping:**

| Audit-Fund | Issue-Typ | Label(s) |
|---|---|---|
| Missing Abstraction (aus `abstractions.md`) | 1 Issue pro Abstraktion | `enhancement` |
| Bug / silent failure (aus `findings-misc.md`) | 1 Issue pro Bug | `bug` |
| API-Inkonsistenz (Gruppe verwandter Paper-cuts) | 1 Sammel-Issue | `enhancement`, ggf. `documentation` |
| Bereits in #60/#61 getrackt | kein neues Issue, nur Kommentar/Link im Audit-Doc | — |

**Issue-Body-Template (ODT-#69-Stil):**

```markdown
## Problem
[Ein Satz, was fehlt / kaputt ist]

## Reproduktion
[Pointer auf konkretes Beispiel im base-project Repo:
src/examples/<file>.c:<line>, oder Stresstest-Metrik]

## Analyse
[Was passiert im Dispatch / welche Zeilen im ODT-Source sind relevant,
mit file:line Verweisen auf OnDeviceTraining/src/...]

## Vorschlag (nur bei enhancement-Issues)
[Signatur-Skizze oder API-Shape]

## Notes
- Gefunden beim Bau von base-project Examples (Pass 1 Audit)
- Related: #60, #61 (falls zutreffend)
- base-project Commit: <hash>
```

**Human-in-Loop-Gate (nicht verhandelbar):**

Issue-Bodies werden zunächst als Markdown-Dateien unter `docs/proposed-issues/<slug>.md` abgelegt. **Kein `gh issue create` ohne explizite Freigabe pro Issue.** User liest drüber, korrigiert Tonalität/Scope/Duplikate, erst dann laufen die `gh issue create`-Calls einzeln (nicht in Schleife, damit Abbruch möglich bleibt).

**Commit-Reihenfolge am Ende von Pass 1:**

1. `docs/odt-userapi-abstractions.md` committen
2. `docs/odt-userapi-findings-misc.md` committen
3. `docs/proposed-issues/` committen (als Review-Paket)
4. Nach User-Freigabe: Issues einzeln filen, URLs in `docs/proposed-issues/README.md` nachtragen
5. Final-Commit mit Issue-URLs

**Übergabe an die andere Claude-Instanz:**

Jedes gefilete Issue ist self-contained: Reproduction-Pointer ins base-project Repo sind ok (public, gepinnt). Commit-Hash wird im Issue-Body verankert, damit die andere Instanz deterministisch den Stand sieht, auf dem die Analyse basierte.

## Zukünftige Arbeiten (nicht im Scope dieses Specs)

- **Audit Pass 2:** ResNet- und DenseNet-Architekturen als neue Examples; APIs für Skip-Connections, Element-wise Add, Dense-Block-Konkatenation auditieren. Ganz andere Komplexitätsstufe — eigener Spec.
- **Hardware-Validierung der MCU-Targets:** Aktuell nicht alle 5 MCU-Targets verfügbar. Cross-Kompilation für alle, Flash+Run für die verfügbaren. Restliche Targets nachziehen, wenn Hardware da ist.
- **CI-Setup:** Weder im Examples- noch im Audit-Scope.

## Annahmen und offene Punkte

- `es-ude/OnDeviceTraining` bleibt während des Audits stabil (kein großes Refactoring, das unsere Traces invalidiert). Falls doch: Commit-Hash der ODT-Version in den Audit-Artefakten pinnen.
- Issue-Labels `bug`, `enhancement`, `documentation` sind im ODT-Repo bereits vorhanden (bestätigt).
- `gh` CLI ist authentifiziert (siehe CLAUDE.md jj+gh Gotchas).
