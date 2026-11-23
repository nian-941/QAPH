# Validation

All eleven synthetic training/prediction checks passed (QAPH RT/TP and nine baselines). Mathematical definitions were compared structurally before and after module extraction. These are code execution checks, not benchmark results.

All eleven integration metric pairs are identical to the previous source layout. All 215 initialized model state tensors and state keys are identical. Prediction-head forward and gradient differences are zero in both eval and train modes with controlled dropout. These checks establish organizational equivalence on the tested inputs, not benchmark accuracy or universal numerical equivalence.

```json
{
  "python": "3.7.16 (default, Jan 17 2023, 16:06:28) [MSC v.1916 64 bit (AMD64)]",
  "packages": {
    "torch": "1.12.0+cu113",
    "torch-geometric": "2.2.0",
    "numpy": "1.21.6",
    "scipy": "1.7.3",
    "pandas": "1.3.5",
    "scikit-learn": "0.24.2",
    "tqdm": "4.67.1",
    "psutil": "7.1.3"
  }
}
```

Run `python tests/smoke_test.py` to repeat.
