# Where to read the implementation

| Component | Source |
|---|---|
| QAPH model assembly / complete forward | `qaph/model.py` |
| CAPH, HAP, adaptive QAPH prediction | `qaph/heads.py` |
| Cauchy-family losses and AGH | `qaph/losses.py`; criterion construction in `qaph/training/engine.py` |
| SEGA graph propagation / residual gating | `qaph/encoders/sega.py` |
| MS-TGEM E/T/F computation | `qaph/encoders/temporal.py` |
| Historical FIG construction and normalization | `qaph/graphs/fig.py` |
| Statistical prior / GMM | `qaph/features/statistics.py` |
| Historical neighbor selection | `qaph/features/neighbors.py` |
| Historical temporal features, anchors, graph-cache refresh and optimizer loop | `qaph/training/engine.py` |
| Historical feature cache and chronological split | `shared/protocol.py` |
| Baseline mask verification | `shared/splits.py` |

Model assembly, prediction heads and losses are separate files. The loss/class bodies are extracted unchanged from frozen v0. Original class names with version suffixes are retained where changing them would obscure source identity; `from qaph import QAPH` is the public model alias.

The training engine retains the native feature builders and optimizer/evaluation loop together to preserve their ordering, random-state behavior and graph-cache semantics. This release organizes the source without introducing a new scientific implementation. Source hash paths and imports are updated explicitly; mathematical definitions are compared structurally, and training/prediction paths are integration-tested.

All baseline-specific models and features live under `baselines/<method>/`. `_shared` contains genuine shared tensor kernels and two-stage factorization utilities. The shared kernels expose method-dependent objectives; the public dispatcher supplies the selected method. The BNLFT/CTF stage split and TRCF historical adaptation remain explicitly documented.

No baseline module imports the QAPH implementation. Shared data code is independent of either model family. The test fixture creates inputs and does not ship fitted artifacts or benchmark configurations.
