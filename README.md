
# Aluminum Price Risk Simulation
### Automotive Engineering Thesis Project - Politecnico di Torino

This project implements a **Monte Carlo Simulation** using the **Euler-Maruyama method** to forecast LME Aluminum prices over a 6-month production horizon.

## Methodology
- **Model:** Geometric Brownian Motion (GBM)
- **Numerical Solver:** Euler-Maruyama (Strong Order 0.5, Weak Order 1.0)
- **Parameters:** Calibrated from 2024-2026 LME historical data.
- **Objective:** Determine the "Probability of Breach" for a budget threshold of $4,380/t.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the simulation: `python aluminum_sim.py`
