# Traffic_prediction
# GRU–LSTM with Temporal Attention for Traffic Flow Prediction

This repository contains MATLAB code for the paper *Advancing Urban Planning with Deep Learning: Intelligent Traffic Flow Prediction and Optimization for Smart Cities*. The code implements a hybrid GRU–LSTM network augmented with a temporal attention mechanism for short-term traffic forecasting on the Caltrans PeMS datasets.

---

## Features
- **Hybrid architecture**: Sequential GRU and LSTM layers for robust temporal modeling.
- **Temporal attention**: Highlights contextually relevant time steps in traffic flow sequences.
- **Data preprocessing**: Normalization, missing value imputation, and train/validation/test partitioning.
- **Flexible loader**: Example provided for the PeMS-04 dataset; easily extensible to PeMS-03, -07, and -08.
- **Evaluation**: Reports MAE, RMSE, and MAPE, along with inference time benchmarking.

---

## Requirements
- MATLAB R2023a or later (tested with MATLAB 2024a).
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html).
- [npy-matlab](https://github.com/kwikteam/npy-matlab) (if loading `.npz`/`.npy` files from PeMS).
- GPU with CUDA support (optional but recommended).

---

## Dataset
We use the **Caltrans PeMS traffic datasets** (PeMS-03, PeMS-04, PeMS-07, PeMS-08).  
- Public datasets are available at: [PeMS Official Site](https://pems.dot.ca.gov/) and via processed `.npz` files from prior benchmarks (e.g., DCRNN, STGCN).  
- Place your dataset files in `./PeMSXX/` folders, where `XX` is the district ID (e.g., `PeMS04/`).

### Expected Files
- `pems04.npz` or `.mat`: traffic flow matrix `[time × sensors]`.
- `adj_pems04.npy`: adjacency matrix of sensor graph.

---

## Code Structure
├── main_pems_gru_lstm_attention.m # Main script 
├── lload_pems_example.m # Loader & preprocessor of data
├── create_sequences.m # Utility to create look-back/horizon samples
├── fit_minmax.m # helper function
├── apply_minmax.m # helper function
├── compute_metrics.m # helper function
├── impute_missing.m # helper function
├── results/ # Folder for saving metrics & figures
└── README.md # This file

Citation: 
If you use this code, please cite our paper:
@article{albalooshi2025urban,
  title   = {Advancing Urban Planning with Deep Learning: Intelligent Traffic Flow Prediction and Optimization for Smart Cities},
  author  = {Albalooshi, Fatema A.},
  journal = {MDPI Future Transportation},
  year    = {2025}
}
