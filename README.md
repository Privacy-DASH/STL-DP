# STL-DP: Differentially Private Time Series Exploring Decomposition and Compression Methods

This repository contains the official implementation for the paper "[STL-DP: Differentially Private Time Series Exploring Decomposition and Compression Methods](https://ceur-ws.org/Vol-3318/short5.pdf)" presented at the CIKM-PAS Workshop 2022.

## Data Preparation

- **Original Data** (no DP perturbation)
  - Run `data_processing.ipynb` to perform time encoding and min-max normalization for each column.
  - Split the data into train:validation:test sets at a 7:1:2 ratio.

- **Perturbed Data** (LPA/FPA/sFPA/tFPA)
  - Use `MakeDatasets_1.Rmd` to inject noise into the Zone 1–3 Power Consumption values.
  - Replace the Zone 1–3 Power Consumption columns with the perturbed data in `data_processing.ipynb` and re-split the data.

## Required Libraries
`numpy`, `pandas`, `torch`, `torchmetrics`, `datetime`

## Measuring Euclidean Distance Between Original and Perturbed Data
Run `Euclidean_dist_2.Rmd`

## Measuring Delta MAPE
Run `run.py`  
Results and logs are saved in the `results/` directory.
