import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

class BayesianOnlineChangePointDetector:
    """
    Simplified Bayesian Online Changepoint Detection
    """
    
    def __init__(self, data, lambda_param=100, max_run_length=None):
        self.data = np.array(data, dtype=float)
        self.T = len(data)
        self.lambda_param = lambda_param
        self.max_run_length = max_run_length or min(self.T, 200)
        
        self.hazard = 1.0 / self.lambda_param
        self.run_length_posterior = np.zeros((self.T, self.max_run_length))
        self.run_length_posterior[0, 0] = 1.0
        
        # Track sufficient statistics
        self.means = np.zeros((self.T, self.max_run_length))
        self.stds = np.ones((self.T, self.max_run_length))
        self.counts = np.zeros((self.T, self.max_run_length))
        
    def fit(self):
        """Run BOCPD"""
        for t in range(1, self.T):
            x = self.data[t]
            max_r = min(t, self.max_run_length - 1)
            
            # Compute likelihood for each run length
            likelihoods = np.zeros(max_r + 1)
            
            for r in range(max_r + 1):
                if r == 0:
                    # New regime: use global mean/std
                    mu = self.data[:t].mean()
                    sigma = self.data[:t].std() + 0.1
                else:
                    # Existing regime: use statistics from this run
                    mu = self.means[t-1, r]
                    sigma = self.stds[t-1, r]
                
                sigma = max(sigma, 0.5)  # Avoid numerical issues
                likelihoods[r] = stats.norm.pdf(x, mu, sigma)
            
            # Update run length posterior using Bayes rule
            # P(r_t | x_1:t) ∝ P(x_t | r_{t-1}) * [H(r)*P(r=0) + (1-H(r))*P(r|r-1)]
            
            # Growth probabilities (no changepoint)
            growth = (1 - self.hazard) * self.run_length_posterior[t-1, :max_r]
            
            # Changepoint probability
            cp_prob = self.hazard * np.sum(self.run_length_posterior[t-1, :max_r+1])
            
            # Posterior
            posterior = np.zeros(max_r + 1)
            posterior[0] = cp_prob * likelihoods[0]
            if max_r > 0:
                posterior[1:max_r+1] = growth * likelihoods[:max_r]
            
            # Normalize
            norm = np.sum(posterior)
            if norm > 0:
                self.run_length_posterior[t, :max_r+1] = posterior / norm
            else:
                self.run_length_posterior[t, 0] = 1.0
            
            # Update sufficient statistics for next step
            for r in range(max_r + 1):
                if r == 0:
                    # New run: just this point
                    self.means[t, 0] = x
                    self.stds[t, 0] = 1.0
                    self.counts[t, 0] = 1
                else:
                    # Continuing run: accumulate
                    prev_mean = self.means[t-1, r-1]
                    prev_count = self.counts[t-1, r-1]
                    prev_std = self.stds[t-1, r-1]
                    
                    new_count = prev_count + 1
                    new_mean = (prev_count * prev_mean + x) / new_count
                    # Simple online variance estimate
                    new_std = np.sqrt((prev_std**2 * prev_count + (x - new_mean)**2) / new_count + 0.1)
                    
                    self.means[t, r] = new_mean
                    self.stds[t, r] = new_std
                    self.counts[t, r] = new_count
    
    def get_changepoint_times(self, threshold=0.5):
        changepoints = []
        for t in range(1, self.T):
            if self.run_length_posterior[t, 0] > threshold:
                changepoints.append(t)
        return changepoints


# Test with simple data
print("Creating synthetic data with changepoint...")
np.random.seed(42)

# Regime 1: mean = 100, std = 10
regime1 = np.random.normal(100, 10, 100)

# Changepoint at week 150
# Regime 2: mean = 140, std = 10 (persistent increase)
regime2 = np.random.normal(140, 10, 150)

# Spike around week 80 (temporary)
spike = np.random.normal(120, 5, 30)

synthetic_data = np.concatenate([regime1[:80], spike, regime1[80:], regime2])

print(f"Data shape: {synthetic_data.shape}")
print(f"Regime 1 mean: {synthetic_data[:100].mean():.2f}")
print(f"Spike mean: {synthetic_data[80:110].mean():.2f}")
print(f"Regime 2 mean: {synthetic_data[200:].mean():.2f}")

# Run detector
detector = BayesianOnlineChangePointDetector(synthetic_data, lambda_param=50, max_run_length=100)
print("\nFitting detector...")
detector.fit()

changepoints = detector.get_changepoint_times(threshold=0.5)
print(f"Detected changepoints: {changepoints}")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Data
ax = axes[0]
ax.plot(synthetic_data, 'o-', alpha=0.6, linewidth=1)
ax.axvline(80, color='blue', linestyle='--', label='Spike (temporary)', linewidth=2)
ax.axvline(180, color='green', linestyle='--', label='True changepoint', linewidth=2)
for cp in changepoints:
    ax.axvline(cp, color='red', alpha=0.5, linestyle=':', linewidth=1)
ax.set_ylabel('Sales Volume')
ax.set_title('Synthetic Data: Temporary Spike vs Real Changepoint')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Changepoint probability
ax = axes[1]
cp_probs = detector.run_length_posterior[:, 0]
ax.plot(cp_probs, color='darkred', linewidth=2)
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold')
ax.fill_between(range(len(cp_probs)), 0, cp_probs, alpha=0.3)
ax.set_ylabel('P(changepoint)')
ax.set_xlabel('Time')
ax.set_title('Changepoint Probability')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('bocpd_demo.png', dpi=100)
print("\nPlot saved to bocpd_demo.png")
plt.show()
