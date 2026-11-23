# QAPH

Source code for QAPH and nine temporal QoS prediction baselines.

## Repository layout

```text
qaph/                  QAPH implementation only
  model.py             Model assembly and forward pass
  heads.py             CAPH, HAP and QAPH prediction heads
  losses.py            Cauchy-family losses and AGH
  encoders/            SEGA and MS-TGEM
  features/            Statistical priors and neighborhood features
  graphs/              Historical FIG and graph construction
  data/                Native QoS loading and dataset support
  training/            Feature construction, training, cache updates and evaluation
  train.py             QAPH command-line entry point
baselines/             Baselines only, one folder per method
  wspred/ nntf/ bnlft/ ctf/ trcf/ gmcl/ hctn/ tpmcf/ gacl/
  _shared/             Shared tensor kernels and two-stage training utilities
  train.py             Unified baseline command-line entry point
shared/                Common data preparation, split validation and I/O
tests/                 Small generated-data integration checks
data/                  Dataset download instructions only
docs/                  Architecture and baseline implementation notes
```

QAPH's core classes are extracted from frozen v0 without changing their computational definitions. The training behavior, including the original SEGA gradient/cache path, is preserved. Namespace and file-provenance paths are updated for this layout. See `SOURCE_MANIFEST.json`, `docs/architecture.md` and `VALIDATION.md`.

## Install and check

Create an isolated Python environment, install a CPU/CUDA PyTorch build for your system, then install the requirements:

```sh
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

Alternatively, use `conda env create -f environment.yml`. The tests generate small inputs and exercise QAPH RT/TP and all nine baselines. They do not download data or reproduce benchmark results. The included GitHub Actions workflow runs the same CPU checks after upload.

## Dataset and shared masks

Obtain WS-DREAM Dataset #2 from its [official download page](https://wsdream.github.io/dataset/wsdream_dataset2.html) or [Zenodo record](https://zenodo.org/records/1133476). Dataset files are not redistributed. See `data/README.md` for the input layout.

Replace uppercase placeholders with values for your own run:

```sh
python -m shared.prepare_data --input /path/to/dataset2/rtdata.txt --qos rt --density DENSITY --seed SEED --output data/prepared/rt
```

The shared mask records ordered training, validation and test identities and validates data hashes. Zero denotes unavailable observations; RT=20 is retained.

## Run QAPH

```sh
python -m qaph.train --qos_type rt --data_dir /path/to/dataset2 --target_slice TARGET --history_length HISTORY_LENGTH --density DENSITY --seed SEED --split_id SEED --gamma GAMMA --epochs EPOCHS --device cpu --log_dir runs/qaph
```

Use `--device cuda:0` for a CUDA run. The QAPH wrapper selects the historical representation and adaptive prediction architecture and final-only lambda-zero evaluation. TP's segmented Cauchy objective is available through `--use_scl` and its native options. Run `python -m qaph.train --help` to inspect the interface.

## Run a baseline

```sh
python -m baselines.train --method METHOD --tensor data/prepared/rt/rt.npy --mask data/prepared/rt/split.npz --output-dir runs/METHOD --epochs EPOCHS --device cpu
```

`METHOD` is one of `wspred`, `nntf`, `bnlft`, `ctf`, `trcf`, `gmcl`, `hctn`, `tpmcf` or `gacl`. QoS type and seed are taken from the mask. Method-specific options are exposed by the dispatcher and native training modules. An existing nonempty output directory is not overwritten.

Baseline implementations and protocol adaptations are described in `docs/baselines.md`. They share evaluation masks and held-out label isolation, while retaining different native inputs and objectives. These are project reproductions/adaptations rather than unmodified original-author releases.

## Release scope

This is a source release with generic run interfaces. Paper-specific experiment recipes, per-run settings, original datasets, trained weights, caches and unpublished logs are not included. Intrinsic model and CLI defaults remain visible and are not advertised as the recipes that produced paper tables.

See `NOTICE.md` and `references.bib` for attribution. This packaging step does not assign a new repository-wide license.
