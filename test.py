import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

class BayesianOnlineChangePointDetector:
    """
    Simplified Bayesian Online Changepoint Detection (Adams & MacKay, 2007)
    Detects sustained baseline shifts while filtering temporary spikes
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
                    # Online variance estimate
                    new_std = np.sqrt((prev_std**2 * prev_count + (x - new_mean)**2) / new_count + 0.1)
                    
                    self.means[t, r] = new_mean
                    self.stds[t, r] = new_std
                    self.counts[t, r] = new_count
    
    def get_changepoint_times(self, threshold=0.5):
        """Get times where changepoint probability exceeds threshold"""
        changepoints = []
        for t in range(1, self.T):
            if self.run_length_posterior[t, 0] > threshold:
                changepoints.append(t)
        return changepoints
    
    def get_changepoint_times_with_confirmation(self, threshold=0.5, lookahead=5, persistence_weeks=3):
        """
        Get changepoint times with delayed confirmation to filter spikes.
        
        Args:
            threshold: P(changepoint) must exceed this
            lookahead: How many weeks to look ahead before confirming
            persistence_weeks: New regime must persist for this many weeks
        """
        changepoints = []
        
        for t in range(1, self.T - lookahead):
            # Step 1: Detect spike/anomaly
            if self.run_length_posterior[t, 0] < threshold:
                continue  # Not an anomaly candidate
            
            # Step 2: Hold judgment, look ahead
            future_window = self.data[t:t+lookahead]
            
            # Step 3: Calculate new regime statistics in lookahead window
            new_mean = np.mean(future_window)
            new_std = np.std(future_window)
            
            # Get old regime stats (from before the anomaly)
            old_mean = self.means[t-1, 1] if t > 1 else self.data[:t].mean()
            old_std = self.stds[t-1, 1] if t > 1 else self.data[:t].std()
            
            # Step 4: Is the new regime actually different AND persistent?
            mean_shift = abs(new_mean - old_mean)
            mean_shift_pct = mean_shift / (old_mean + 0.1)
            
            # Check if shift is significant (e.g., >10% change)
            is_significant = mean_shift_pct > 0.10
            
            # Check if new regime persists (stays away from old mean)
            persistence_count = 0
            for i in range(t, min(t + persistence_weeks, self.T)):
                if abs(self.data[i] - old_mean) > old_std:
                    persistence_count += 1
            
            is_persistent = persistence_count >= persistence_weeks - 1
            
            # Step 5: Confirm changepoint only if both conditions met
            if is_significant and is_persistent:
                changepoints.append(t)
                # Skip ahead to avoid overlapping detections
                t += lookahead
    
        return changepoints

    def get_regime_characteristics(self, t):
        """
        Get the mean and std of the current regime at time t.
        Helps you understand what regime you're in.
        """
        run_lengths = np.arange(self.max_run_length)
        posterior = self.run_length_posterior[t, :]
        
        # Weighted average of regime means
        regime_mean = np.sum(posterior * self.means[t, :])
        regime_std = np.sum(posterior * self.stds[t, :])
        
        return regime_mean, regime_std

# Load data
print("Loading sales data...")
df = pd.read_csv("synthetic_retail_sales_many_spikes_500_weeks.csv", encoding='ISO-8859-1')
sales_data = df['sales_volume'].values

print(f"Data loaded: {len(sales_data)} weeks")
print(f"Sales range: {sales_data.min():.2f} - {sales_data.max():.2f}\n")

# Run Bayesian Online Changepoint Detection
# lambda_param: expected run length (weeks between changepoints)
# Larger lambda = fewer false positives (expects fewer changepoints)
print("Running Bayesian Online Changepoint Detection...")
detector = BayesianOnlineChangePointDetector(sales_data, lambda_param=150, max_run_length=200)
detector.fit()

# Get detected changepoints
changepoints = detector.get_changepoint_times_with_confirmation(
    threshold=0.5,
    lookahead=5,              # Look ahead 5 weeks
    persistence_weeks=3       # Must be different for 3+ weeks
)
print(f"Detected {len(changepoints)} changepoint events")
if len(changepoints) > 0:
    print(f"First 10 changepoint weeks: {changepoints[:10]}\n")

# Create visualizations with non-GUI backend
plt.switch_backend('Agg')
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Sales data with detected changepoints
ax = axes[0]
ax.plot(sales_data, 'o-', color='black', alpha=0.6, linewidth=1.5, label='Weekly Sales', markersize=3)
for i, cp in enumerate(changepoints[:15]):  # Show first 15
    ax.axvline(cp, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax.set_ylabel('Sales Volume', fontsize=12)
ax.set_title('Weekly Sales Data with Detected Changepoints (Red Lines)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Changepoint probability over time
ax = axes[1]
cp_probs = detector.run_length_posterior[:, 0]
ax.plot(cp_probs, color='darkred', linewidth=2, label='P(changepoint at t)')
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Decision threshold (0.5)', linewidth=2)
ax.fill_between(range(len(cp_probs)), 0, cp_probs, alpha=0.3, color='red')
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Changepoint Probability Over Time', fontsize=14)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Run length posterior (heatmap)
# Darker colors at low run lengths = recent changepoint detected
ax = axes[2]
rlp_display = detector.run_length_posterior[:, :min(100, detector.max_run_length)].T
im = ax.imshow(rlp_display, aspect='auto', origin='lower', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Run Length (weeks since change)', fontsize=12)
ax.set_title('Run Length Posterior Heatmap (Darker = More Likely Changepoint)', fontsize=14)
cbar = plt.colorbar(im, ax=ax, label='Posterior Probability')

plt.tight_layout()
plt.savefig('changepoint_detection.png', dpi=100, bbox_inches='tight')
print("Visualization saved to changepoint_detection.png")
plt.close()

# Summary statistics
print(f"\n--- Results Summary ---")
print(f"Max changepoint probability: {cp_probs.max():.4f}")
print(f"Number of weeks with high changepoint probability: {len(changepoints)}")
if len(changepoints) > 1:
    gaps = np.diff(changepoints)
    print(f"Mean gap between changepoint signals: {np.mean(gaps):.1f} weeks")
    print(f"Median gap: {np.median(gaps):.1f} weeks")
    
print("\n✅ Analysis complete! Check 'changepoint_detection.png' for visualization.")