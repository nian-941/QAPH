# Baseline implementations

| Method | Implementation in this release |
|---|---|
| WSPred | Tensor factorization with pair-mean regularization; historical observations and target-training observations may enter its native objective. |
| NNTF | Nonnegative tensor factorization with the literal dense zero-filled reconstruction denominator. |
| BNLFT | Biased nonnegative factorization; history-only unsupervised pretraining followed by target-training-only fitting of copied factors. |
| CTF | Cauchy tensor factorization with history-only unsupervised pretraining followed by target-training-only fitting. |
| TRCF | Historical-endpoint adaptation (`trcf_h` in the experiment engineering code); constructs similarity/deviation evidence from history and fits a global mixture on target-training labels. |
| GMCL | Structured graph propagation and contrastive learning adaptation; its graph may use history and target-training observations. |
| HCTN | Independent hypergraph-convolution/Transformer reproduction with historical inputs. |
| TPMCF | PyTorch paper-aligned adaptation with collaborative features, graph-convolution feature extraction and temporal prediction; not the original TensorFlow program. |
| GACL | Graph Attention Collaborative Learning reproduction with historical inputs and explicit neighbor sampling; not a graph-contrastive method. |

All methods select checkpoints using validation data and evaluate held-out test labels at the final stage. Only positive targets are scored; RT=20 is kept. Method-native historical reconstruction or target-training graph access is not described as identical input information across methods.

Original algorithm kernels are retained. Public launchers replace machine-specific batch controllers and private manifest dependencies. The QAPH and TPMCF entry points additionally permit an explicitly selected CPU device; CUDA timing/memory calls are guarded. No model or objective is changed. The public two-stage launcher exposes the history initialization seed as the run seed; exact experiment-specific history-seed choices are not bundled as a paper recipe.
