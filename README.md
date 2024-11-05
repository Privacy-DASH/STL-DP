# STL-DP: Differentially Private Time Series Exploring Decomposition and Compression Methods

![STL-DP](https://github.com/user-attachments/assets/101fd014-8736-4928-bff5-55c98e8d170a)

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
![Table1](https://github.com/user-attachments/assets/9fb6760e-1d3f-4f19-90d8-40932e83887c)

## Measuring Delta MAPE
Run `run.py`  
Results and logs are saved in the `results/` directory.
![Table2](https://github.com/user-attachments/assets/58811e2a-5c89-4a65-9fdc-c1cb2b0ea5ba)
