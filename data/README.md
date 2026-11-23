# WS-DREAM Dataset #2

Official download page: https://wsdream.github.io/dataset/wsdream_dataset2.html

Provider-hosted archive: https://zenodo.org/records/1133476

The time-aware dataset contains 142 users, 4,500 services and 64 time slices. Obtain `dataset2/rtdata.txt` and `dataset2/tpdata.txt` from the provider's archive.

The loaders expect four whitespace-separated columns without a header:

```text
user_id service_id time_index qos_value
```

IDs are zero-based. Tensors use `[user, service, time]`; public CLI target-slice numbers are one-based. Keep the original QoS units and values, including RT=20. Unavailable records are represented by zero; negative or nonfinite input values are rejected.

`prepare_data.py` accepts an explicit shape for other datasets; the original WS-DREAM shape is its default. It builds the tensor, creates ordered disjoint target masks, and records content hashes. The baseline loader checks those masks against the tensor and seeded order before training. No dataset is included in the repository.
