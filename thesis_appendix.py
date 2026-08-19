import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import invgauss, norm

# 1. Simulation Setup and Random Number Generator Seed
SEED = 100          
rng  = np.random.default_rng(SEED)

# 2. Calibrated Input Parameters
mu        = 0.192393444       # Calibrated annual drift (19.24%)
sigma     = 0.193560675       # Calibrated annual volatility (19.36%)
S0        = 3480.70           # Baseline spot price from April 10th, 2026 ($/t)
T         = 0.5               # 6-month operational horizon (in years)
N         = 126               # Daily steps (trading days per horizon)
M         = 10000             # Monte Carlo trajectories
threshold = 4380.185941       # Industrial budget break-even ceiling (K)

dt = T / N                    # Temporal step increment

# 3. Euler-Maruyama Path Generation
dW = np.sqrt(dt) * rng.standard_normal((M, N))   
S  = np.zeros((M, N + 1))                     
S[:, 0] = S0 

for j in range(N):
    S[:, j + 1] = S[:, j] + mu * S[:, j] * dt + sigma * S[:, j] * dW[:, j] 

# 4. Verification and Exact Analytic Solution
W_paths      = np.cumsum(dW, axis=1)    
t_grid       = np.linspace(dt, T, N)      
S_exact      = S0 * np.exp((mu - 0.5 * sigma**2) * t_grid + sigma * W_paths)

# 5. Core Metric Extraction
terminal_prices  = S[:, -1]                        
simulated_mean   = np.mean(terminal_prices)
theoretical_T    = S0 * np.exp(mu * T)

weak_error = abs(simulated_mean - theoretical_T)
p_breach    = np.mean(terminal_prices > threshold)
any_breach  = np.mean(np.any(S > threshold, axis=1))

print(f"Theoretical E[S(T)]:  ${theoretical_T:,.2f}/t")
print(f"Simulated mean S(T):  ${simulated_mean:,.2f}/t")
print(f"Terminal Breach P:    {p_breach:.2%}")
print(f"Path-Dependent P:     {any_breach:.2%}")