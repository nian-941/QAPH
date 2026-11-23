# Source and attribution

QAPH code is drawn from the project's frozen-v0 source snapshot. Original source headers are preserved. Only the rolling CLI's GPU-only device guard is relaxed to permit an explicitly requested CPU run; model equations, gradient/cache behavior, losses and training logic are unchanged. Historical development comments and compatibility options in that snapshot are not claims of additional evaluated model variants.

Baseline code consists of project reproductions, equation-based ports and protocol adaptations. Cite the original papers in `references.bib` as well as the QAPH article when using this work. In particular, the CTF factorization kernel was developed with the authors' released C++ implementation as a reference; TRCF, TPMCF and GACL adaptations are described explicitly in `docs/baselines.md`.

The packaging process does not assign a new repository-wide license or replace any existing rights/attribution in source. Dataset access and reuse follow the original dataset providers' terms. No third-party dataset files or model weights are redistributed here.
