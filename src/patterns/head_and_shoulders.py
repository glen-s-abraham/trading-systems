import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# If needed, adjust your paths
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from utils import YfinanceUtils
# from utils import PdUtils

#######################################
# Parameters and Utilities
#######################################

SYMBOL = "ITC.NS"
START_DATE = "2019-12-19"
END_DATE = "2024-12-19"

# Pattern detection parameters
SHOULDER_TOL = 0.02   # Shoulders height tolerance relative to head height
MAX_SEP = 30          # Maximum separation in indices between left shoulder and right shoulder
SMOOTH_WINDOW = 5     # Moving average window for smoothing the price data

#######################################
# Data Retrieval - Example
#######################################

# If you have your own data loading utilities, use them; otherwise:
# Here we'll use yfinance directly for demonstration
try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    sys.exit(1)

df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
if df.empty:
    print("No data retrieved. Check the symbol and date range.")
    sys.exit(1)

# Use Adj Close for pattern detection and plotting
df['Adj Close SMA'] = df['Adj Close'].rolling(SMOOTH_WINDOW).mean().fillna(method='bfill')

price = df['Adj Close SMA'].values
time = np.arange(len(df))

#######################################
# Peak Detection
#######################################

# Find peaks in the smoothed Adj Close
# Adjust parameters like distance, prominence, etc., to get better peak detection
peaks, properties = find_peaks(price, distance=5, prominence=1)

# peaks is an array of indices where local maxima occur

#######################################
# Head and Shoulders Detection Logic
#######################################

found_patterns = []

# We need at least 3 peaks to form a pattern
if len(peaks) >= 3:
    # We will try every consecutive triple of peaks
    for i in range(len(peaks)-2):
        ls_idx = peaks[i]
        h_idx = peaks[i+1]
        rs_idx = peaks[i+2]

        ls_val = price[ls_idx]
        h_val = price[h_idx]
        rs_val = price[rs_idx]

        # Check that the head is the highest
        if h_val <= ls_val or h_val <= rs_val:
            continue

        # Check that the shoulders are roughly the same height
        # Shoulders should be close in height: abs(ls_val - rs_val) <= SHOULDER_TOL * h_val
        if abs(ls_val - rs_val) > SHOULDER_TOL * h_val:
            continue

        # Check spacing: the pattern should not be too stretched out
        if (rs_idx - ls_idx) > MAX_SEP:
            continue

        # If criteria are met, we call this a head and shoulders pattern
        found_patterns.append((ls_idx, h_idx, rs_idx))

#######################################
# Plotting
#######################################

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time, df['Adj Close'], label='Adj Close Price', color='blue', alpha=0.7)
ax.plot(time, price, label=f'Adj Close SMA({SMOOTH_WINDOW})', color='orange', alpha=0.9)

if found_patterns:
    # Take the first found pattern for demonstration
    ls, h, rs = found_patterns[0]

    # Mark the shoulders and head
    ax.scatter(time[ls], price[ls], color='green', s=100, label='Left Shoulder')
    ax.scatter(time[h], price[h], color='red', s=100, label='Head')
    ax.scatter(time[rs], price[rs], color='green', s=100, label='Right Shoulder')

    # Draw dashed lines connecting peaks
    ax.plot([time[ls], time[h]], [price[ls], price[h]], 'k--')
    ax.plot([time[h], time[rs]], [price[h], price[rs]], 'k--')

    # Draw neckline
    # We'll approximate the neckline by finding the lowest valley between LS and H, and between H and RS
    left_valley_range = range(ls, h)
    right_valley_range = range(h, rs)

    left_valley = min(left_valley_range, key=lambda x: price[x])
    right_valley = min(right_valley_range, key=lambda x: price[x])

    ax.plot([time[left_valley], time[right_valley]],
            [price[left_valley], price[right_valley]],
            color='purple', linestyle=':', label='Neckline')

    ax.set_title("Head and Shoulders Pattern Detected")
else:
    ax.set_title("No Head and Shoulders Pattern Detected")

ax.set_xlabel("Bars (Index)")
ax.set_ylabel("Price")
ax.legend()
plt.grid(True)
plt.show()
