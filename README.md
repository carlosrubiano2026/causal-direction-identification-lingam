# LiNGAM vs Mean-Independence Causal Direction Simulation

This project simulates a simple linear causal model to compare two methods for recovering causal direction:

1. DirectLiNGAM
2. A mean-independence residual-based heuristic

The simulation compares performance under Gaussian and non-Gaussian data-generating processes across different sample sizes.

## Main idea

The true data-generating process is:

Y = beta X + epsilon

so the correct causal direction is:

X -> Y

The experiment evaluates how often each method recovers the correct direction.

## Repository structure

- `notebooks/Simulation.ipynb`: exploratory notebook
- `src/simulation.py`: reproducible Python script
- `outputs/figures/`: generated plots
- `outputs/tables/`: summary results

## How to run

```bash
pip install -r requirements.txt
python src/simulation.py
