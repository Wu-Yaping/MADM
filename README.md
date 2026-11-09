# Supplementary Code for the Submitted Paper

Implementation of the proposed MADM model for multivariate time series anomaly detection.

## Requirements

```
Python 3.9+
pip install -r requirements.txt
```

## Datasets

This code supports five public benchmarks: **PSM, MSL, SMAP, SMD, SWaT**.

Download links:
- PSM, MSL, SMAP: [Google Drive](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR)
- SMD: [GitHub](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
- SWaT: [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) (requires registration)

## Dataset Directory Structure

Place the downloaded files under `./dataset/` as follows:

```
dataset/
  PSM/
    train.csv
    test.csv
    test_label.csv
  MSL/
    MSL_train.npy
    MSL_test.npy
    MSL_test_label.npy
  SMAP/
    SMAP_train.npy
    SMAP_test.npy
    SMAP_test_label.npy
  SMD/
    SMD_train.npy
    SMD_test.npy
    SMD_test_label.npy
  SWAT/
    SWAT_train.npy
    SWAT_test.npy
    SWAT_test_label.npy
```

## Training and Evaluation

```bash
# Train
python main.py --mode train --dataset PSM --data_path ./dataset/PSM/ --input_c 25

# Test (loads the saved checkpoint)
python main.py --mode test --dataset PSM --data_path ./dataset/PSM/ --input_c 25
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | PSM | Dataset name: PSM, MSL, SMAP, SMD, SWaT |
| `--input_c` | 25 | Number of sensors (PSM=25, MSL=55, SMAP=25, SMD=38, SWaT=51) |
| `--win_size` | 100 | Sliding window length |
| `--k` | 0.5 | Temporal discrepancy coefficient |
| `--lambda_r` | 0.05 | Spatial discrepancy coefficient |
| `--k_hypergraph` | 3 | Top-K neighbors for hypergraph construction |
