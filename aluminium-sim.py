"""
GBM Simulation — LME Aluminum Price Risk Model


Model:   dS = mu*S*dt + sigma*S*dW   (Geometric Brownian Motion)
Euler-Maruyama (EM) discretisation
Asset:   LME Aluminum (MAL3)
Time-frame:  6-month horizon (i.e.126 trading days)

Parameters (calibrated on 507 daily observations, 24 months):
    mu    = 0.17   (annual drift,      17%)
    sigma = 0.19   (annual volatility, 19%)
    S0    = 2400   (USD/t, approximate current spot)

Break-even threshold: $4380.185941/t
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import invgauss
from scipy.stats import invgauss, norm

# ── 0/ Simulation set-up ────────────────────────────────────────────────────────
SEED = 100          # mirrors Higham's randn('state', 100)
rng  = np.random.default_rng(SEED)

# ── 1/ Parameters ─────────────────────────────────────────────────────────────
mu        = 0.192393444       # annual drift
sigma     = 0.193560675    # annual volatility
S0        = 3480.70       # initial spot price  (USD/t)
T         = 0.5      # horizon (in years => 6 months)
N         = 126      # daily steps         (≈ trading days in 6 months)
M         = 10000     # number of Monte Carlo paths
threshold = 4380.185941       # break-even ceiling  (USD/t)

dt = T / N              # step increment in years

# ── 2/ Euler-Maruyama simulation ──────────────────────────────────────────────
dW = np.sqrt(dt) * rng.standard_normal((M, N))   
S  = np.zeros((M, N + 1))                     
S[:, 0] = S0 

for j in range(N):
    S[:, j + 1] = S[:, j] + mu * S[:, j] * dt + sigma * S[:, j] * dW[:, j] 

# ── 3/ Exact GBM solution (for validation) ────────────────────────────────────
W_paths      = np.cumsum(dW, axis=1)    
t_grid       = np.linspace(dt, T, N)      
S_exact      = S0 * np.exp((mu - 0.5 * sigma**2) * t_grid + sigma * W_paths)

# ── 4/ Theoretical mean E[S(t)] = S0 * exp(mu * t) ───────────────────────────
t_full       = np.linspace(0, T, N + 1)
theoretical_mean = S0 * np.exp(mu * t_full)

# ── 5/ Key statistics ─────────────────────────────────────────────────────────
terminal_prices  = S[:, -1]                        
simulated_mean   = np.mean(terminal_prices)
theoretical_T    = S0 * np.exp(mu * T)

weak_error = abs(simulated_mean - theoretical_T)
p_breach    = np.mean(terminal_prices > threshold)
any_breach  = np.mean(np.any(S > threshold, axis=1))

# ── 5b/ Advanced Path-Dependent Metrics Extraction ───────────────────────────
max_prices = np.max(S, axis=1)

breach_indices = []
for path in S:
    crossings = np.where(path > threshold)[0]
    if len(crossings) > 0:
        breach_indices.append(crossings[0])
breach_days = np.array(breach_indices)

print("=" * 55)
print("  GBM SIMULATION — LME ALUMINUM  ")
print("=" * 55)
print(f"  Parameters:         mu={mu:.0%}, sigma={sigma:.0%}, S0=${S0:,.0f}")
print(f"  Horizon:            {T:.1f} yr  ({N} trading days)")
print(f"  Paths simulated:    {M:,}")
print("-" * 55)
print(f"  Theoretical E[S(T)]:  ${theoretical_T:,.2f}/t")
print(f"  Simulated mean S(T):  ${simulated_mean:,.2f}/t")
print(f"  Weak error:           ${weak_error:,.2f}  "
      f"({'good' if weak_error < 20 else 'check seed/M'})")
print("-" * 55)
print(f"  Break-even threshold: ${threshold:,.0f}/t")
print(f"  P[S(T) > $4,380]:     {p_breach:.2%}   (terminal breach)")
print(f"  P[any breach]:        {any_breach:.2%}   (path-dependent)")
print("=" * 55)

# ── 6. Plots ──────────────────────────────────────────────────────────────────

# ═ PLOT A: Spaghetti Chart ════════════════════════════════════════════════════
plt.figure(1, figsize=(8, 6)) 
plt.plot(t_full, S[:200, :].T, color='steelblue', alpha=0.06, linewidth=0.6)
plt.plot(t_full, np.mean(S, axis=0), color='steelblue', linewidth=2, label='Simulated mean (10k paths)')
plt.plot(t_full, theoretical_mean, color='navy', linewidth=2, linestyle='--', label=f'Theoretical mean $S_0 e^{{\\mu t}}$')
plt.axhline(threshold, color='crimson', linewidth=1.5, linestyle='-.', label=f'Break-even ${threshold:,.0f}/t')
plt.xlabel("Time (years)", fontsize=11)
plt.ylabel("LME Aluminum price (USD/t)", fontsize=11)
plt.title("Monte Carlo paths — Euler-Maruyama", fontsize=12, fontweight='bold')
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend(fontsize=9, loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("spaghetti_plot.png", dpi=300)

# ═ PLOT B: Terminal Price Distribution ═══════════════════════════════════════
plt.figure(2, figsize=(8, 6)) 
plt.hist(terminal_prices, bins=80, color='steelblue', edgecolor='none', alpha=0.75, density=True, label='S(T) distribution')
plt.axvline(threshold, color='crimson', linewidth=1.8, linestyle='-.', label=f'Break-even ${threshold:,.0f}/t')
plt.axvline(simulated_mean, color='navy', linewidth=1.8, linestyle='--', label=f'Mean ${simulated_mean:,.0f}/t')
plt.axvspan(threshold, terminal_prices.max(), alpha=0.12, color='crimson')
plt.text(threshold + 30, plt.gca().get_ylim()[1] * 0.85, f'P[breach] = {p_breach:.2%}', color='crimson', fontsize=10, va='top')
plt.xlabel("Terminal price S(T) (USD/t)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.title("Distribution of S(T) at 6-month horizon", fontsize=12, fontweight='bold')
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend(fontsize=9, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("histogram_plot.png", dpi=300)

# ═ PLOT C: Maximum Intra-Horizon Price Distribution ══════════════════════════
plt.figure(3, figsize=(8, 6))
plt.hist(max_prices, bins=80, color='darkslateblue', edgecolor='none', alpha=0.75, density=True, label='Max price distribution')
plt.axvline(threshold, color='crimson', linewidth=1.8, linestyle='-.', label=f'Break-even ${threshold:,.0f}/t')
plt.axvline(S0, color='black', linewidth=1.5, linestyle=':', label=f'Spot Price $S_0$ ${S0:,.0f}/t')
plt.axvspan(threshold, max_prices.max(), alpha=0.12, color='crimson')
plt.text(threshold + 30, plt.gca().get_ylim()[1] * 0.85, f'P[any breach] = {any_breach:.2%}', color='crimson', fontsize=10, va='top')
plt.xlabel("Maximum Intra-Horizon Price $\\max(S_t)$ (USD/t)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.title("Distribution of Maximum Price Peaks Across Horizon", fontsize=12, fontweight='bold')
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend(fontsize=9, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("max_price_distribution.png", dpi=300)

# ═ PLOT D: First Passage Time (FPT) Distribution ═════════════════════════════
plt.figure(4, figsize=(8, 6))
if len(breach_days) > 0:
    plt.hist(breach_days, bins=40, color='dodgerblue', edgecolor='none', alpha=0.70, density=True, label='Observed FPT')
    
    # Analytical Inverse Gaussian (Wald) distribution fitting over the trading days
    shape, loc, scale = invgauss.fit(breach_days)
    x_range = np.linspace(0, N, 500)
    pdf_fitted = invgauss.pdf(x_range, shape, loc, scale)
    plt.plot(x_range, pdf_fitted, color='navy', linewidth=2, label='Fitted Inverse Gaussian PDF')

plt.xlabel("Trading Day of First Breach ($t$)", fontsize=11)
plt.ylabel("Density of First Breaches", fontsize=11)
plt.title("First Passage Time (FPT) Density Given Breach", fontsize=12, fontweight='bold')
plt.xlim(0, N)
plt.legend(fontsize=9, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("first_passage_time.png", dpi=300)

# ═ PLOT E: Q-Q Plot — Log-Normality Validation ═══════════════════════════════
# ln(S_T / S_0) ~ N( (mu - 0.5*sigma^2)*T , sigma^2*T )  from Section 2.3.3
from scipy.stats import norm   # ensure norm is available

mu_ln  = (mu - 0.5 * sigma**2) * T    # theoretical log-mean
sig_ln = sigma * np.sqrt(T)            # theoretical log-std

log_returns = np.log(terminal_prices / S0)   # empirical log-returns

n_q                   = len(log_returns)
probabilities         = np.arange(1, n_q + 1) / (n_q + 1)
empirical_quantiles   = np.sort(log_returns)
theoretical_quantiles = norm.ppf(probabilities, loc=mu_ln, scale=sig_ln)

sim_mu_ln  = np.mean(log_returns)
sim_sig_ln = np.std(log_returns)

plt.figure(5, figsize=(7, 7))
plt.scatter(theoretical_quantiles, empirical_quantiles,
            s=0.8, alpha=0.35, color='steelblue', label='Simulated quantiles')

lim_low  = min(theoretical_quantiles.min(), empirical_quantiles.min())
lim_high = max(theoretical_quantiles.max(), empirical_quantiles.max())
plt.plot([lim_low, lim_high], [lim_low, lim_high],
         color='crimson', linewidth=1.8, linestyle='--',
         label='Perfect log-normal (45 degree line)')

plt.xlabel('Theoretical quantiles  N( (mu - 0.5*sigma^2)*T,  sigma^2*T )', fontsize=10)
plt.ylabel('Empirical log-return quantiles  ln(S_T / S_0)', fontsize=10)
plt.title('Q-Q Plot: Log-Normality Validation of Terminal Prices\nLME Aluminium GBM Simulation',
          fontsize=11, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(alpha=0.3)

plt.text(0.04, 0.96,
         f'Theoretical:  mu_ln={mu_ln:.4f},  sigma_ln={sig_ln:.4f}\n'
         f'Simulated:    mu_ln={sim_mu_ln:.4f},  sigma_ln={sim_sig_ln:.4f}',
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig("qq_plot_lognormality.png", dpi=300)
print("Q-Q plot saved.")

# ── 7. Sensitivity table (stress-test sigma) ──────────────────────────────────
print(f"\n  Sensitivity: P[S(T) > ${threshold:,.0f}] under different volatilities")
print("  " + "-" * 35)
print(f"  {'sigma':>8}  {'P[terminal]':>13}  {'P[any breach]':>15}  {'Mean S(T)':>12}")
print("  " + "-" * 55)

for sig_test in [0.10, 0.15, 0.19, 0.25, 0.30, 0.40]:
    dW_t   = np.sqrt(dt) * rng.standard_normal((M, N))
    S_test = np.zeros((M, N + 1))
    S_test[:, 0] = S0
    for j in range(N):
        S_test[:, j + 1] = (S_test[:, j]
                             + mu * S_test[:, j] * dt
                             + sig_test * S_test[:, j] * dW_t[:, j])
    p_term = np.mean(S_test[:, -1] > threshold)
    p_any  = np.mean(np.any(S_test > threshold, axis=1))
    m      = np.mean(S_test[:, -1])
    print(f"  {sig_test:>7.0%}  {p_term:>13.2%}  {p_any:>15.2%}  ${m:>10,.0f}")

print("  " + "-" * 35)

print(f"ACTUAL TERMINAL STD DEV: ${np.std(S[:, -1]):.2f}")
print(f"ACTUAL MEAN PATHWISE STRONG ERROR: ${np.mean(np.abs(S[:, -1] - S_exact[:, -1])):.2f}")

# ── 5c/ Strong Convergence Verification (Shared Brownian Path) ───────────────
# S[:, -1] is the discrete EM terminal price array
# S_exact[:, -1] is the exact analytical terminal price array
# Both arrays share the exact same underlying Wiener increments (dW)

absolute_endpoint_discrepancies = np.abs(S[:, -1] - S_exact[:, -1])
computed_strong_error = np.mean(absolute_endpoint_discrepancies)

print(f"  Mean Pathwise Endpoint Strong Error: ${computed_strong_error:.2f}/t")
print(f"  Theoretical EM Strong Order Match:  O(dt^0.5) ≈ {np.sqrt(dt):.4f}")
print("=" * 55)


# This will pop up all 4 windows simultaneously
plt.show()
plt.close('all')
#python "/Users/maximedejong/Desktop/Polito/Year 3/Polito Thesis/Thesis Simulation/Aluminium-Price-Risk-Simulation/aluminium-sim.py"