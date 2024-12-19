import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

#######################################
# Parameters
#######################################
SYMBOL = "TATAPOWER.NS"
START = "2022-12-19"
END = "2024-12-19"

SMOOTH_WINDOW = 5         # Smoothing window for smoothing the price
PEAK_PROMINENCE = 1       # Adjust if too many peaks are detected or too few
PEAK_DISTANCE = 5         # Minimum distance between peaks
TOLERANCE = 0.01          # Similarity tolerance (~1%)
MIN_DEPTH_RATIO = 0.005   # Relative difference required for pattern shape
TREND_LOOKBACK = 30        # Bars to check for trend before pattern
CONFIRMATION_LOOKFORWARD = 10  # Bars to confirm breakout after pattern forms

#######################################
# Data Loading
#######################################
df = yf.download(SYMBOL, start=START, end=END, interval="1m")
if df.empty:
    print("No data retrieved. Check symbol or date range.")
    exit()

df['Adj Close SMA'] = df['Adj Close'].rolling(SMOOTH_WINDOW).mean().fillna(method='bfill')
price = df['Adj Close SMA'].values
time = np.arange(len(df))

#######################################
# Utility Functions
#######################################
def is_uptrend(prices):
    X = np.arange(len(prices)).reshape(-1,1)
    y = prices
    model = LinearRegression().fit(X, y)
    return model.coef_[0] > 0

def is_downtrend(prices):
    X = np.arange(len(prices)).reshape(-1,1)
    y = prices
    model = LinearRegression().fit(X, y)
    return model.coef_[0] < 0

#######################################
# Pattern Detection Functions
#######################################
def detect_double_tops(price, distance, prominence, tolerance, depth_ratio):
    peaks, _ = find_peaks(price, distance=distance, prominence=prominence)
    patterns = []

    for i in range(len(peaks)-1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        h1 = price[p1]
        h2 = price[p2]

        # Similar peak heights
        if abs(h1 - h2) > tolerance * h1:
            continue

        # Find trough between p1 and p2
        mid_segment = price[p1:p2+1]
        trough_rel = np.argmin(mid_segment)
        trough = p1 + trough_rel
        trough_val = price[trough]

        # Trough sufficiently lower than peaks
        peak_min = min(h1, h2)
        if peak_min - trough_val < depth_ratio * peak_min:
            continue

        # Uptrend before p1
        if p1 > TREND_LOOKBACK:
            prev_segment = price[p1 - TREND_LOOKBACK:p1]
            if not is_uptrend(prev_segment):
                continue
        else:
            continue

        # Confirm pattern by breakout below trough_val after p2
        confirm = False
        look_ahead = min(p2 + CONFIRMATION_LOOKFORWARD, len(price))
        for j in range(p2+1, look_ahead):
            if price[j] < trough_val:
                confirm = True
                break
        if not confirm:
            continue

        # Confirmed double top pattern
        patterns.append((p1, trough, p2))
    return patterns

def detect_double_bottoms(price, distance, prominence, tolerance, depth_ratio):
    inv_price = -price
    troughs, _ = find_peaks(inv_price, distance=distance, prominence=prominence)
    patterns = []
    for i in range(len(troughs)-1):
        t1 = troughs[i]
        t2 = troughs[i+1]
        h1 = price[t1]
        h2 = price[t2]

        # Similar trough depths
        if abs(h1 - h2) > tolerance * abs(h1):
            continue

        # Find peak between t1 and t2
        mid_segment = price[t1:t2+1]
        peak_rel = np.argmax(mid_segment)
        peak = t1 + peak_rel
        peak_val = price[peak]

        # Peak sufficiently above troughs
        trough_max = max(h1, h2)
        if peak_val - trough_max < depth_ratio * trough_max:
            continue

        # Downtrend before t1
        if t1 > TREND_LOOKBACK:
            prev_segment = price[t1 - TREND_LOOKBACK:t1]
            if not is_downtrend(prev_segment):
                continue
        else:
            continue

        # Confirm pattern by breakout above peak_val after t2
        confirm = False
        look_ahead = min(t2 + CONFIRMATION_LOOKFORWARD, len(price))
        for j in range(t2+1, look_ahead):
            if price[j] > peak_val:
                confirm = True
                break
        if not confirm:
            continue

        patterns.append((t1, peak, t2))
    return patterns

#######################################
# Detect Patterns
#######################################
double_top_patterns = detect_double_tops(price, PEAK_DISTANCE, PEAK_PROMINENCE, TOLERANCE, MIN_DEPTH_RATIO)
double_bottom_patterns = detect_double_bottoms(price, PEAK_DISTANCE, PEAK_PROMINENCE, TOLERANCE, MIN_DEPTH_RATIO)

#######################################
# Plotting
#######################################
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time, df["Adj Close"], label="Adj Close Price", alpha=0.7, color='blue')
ax.plot(time, price, label=f"SMA({SMOOTH_WINDOW})", alpha=0.9, color='orange')

# Plot double tops
for (p1, trough, p2) in double_top_patterns:
    # Pattern range: from p1 to p2
    pattern_range = range(p1, p2+1)

    # Shade the entire pattern area
    ax.fill_between(time[pattern_range], price[pattern_range], color='gray', alpha=0.2)
    
    # Draw neckline: horizontal line at trough_val
    trough_val = price[trough]
    ax.axhline(y=trough_val, xmin=time[p1]/time[-1], xmax=time[p2]/time[-1], color='purple', linestyle=':')

    # Mark the main points
    ax.scatter(time[p1], price[p1], color='red', s=100, label='_nolegend_')
    ax.scatter(time[p2], price[p2], color='red', s=100, label='_nolegend_')
    ax.scatter(time[trough], price[trough], color='green', s=100, label='_nolegend_')
    # Connect main points
    ax.plot([time[p1], time[trough], time[p2]], [price[p1], price[trough], price[p2]], 'k--', alpha=0.7)

# Plot double bottoms
for (t1, peak, t2) in double_bottom_patterns:
    # Pattern range: from t1 to t2
    pattern_range = range(t1, t2+1)

    # Shade the entire pattern area
    ax.fill_between(time[pattern_range], price[pattern_range], color='yellow', alpha=0.2)
    
    # Draw neckline: horizontal line at peak_val
    peak_val = price[peak]
    ax.axhline(y=peak_val, xmin=time[t1]/time[-1], xmax=time[t2]/time[-1], color='purple', linestyle=':')

    # Mark the main points
    ax.scatter(time[t1], price[t1], color='green', s=100, label='_nolegend_')
    ax.scatter(time[t2], price[t2], color='green', s=100, label='_nolegend_')
    ax.scatter(time[peak], price[peak], color='red', s=100, label='_nolegend_')
    # Connect main points
    ax.plot([time[t1], time[peak], time[t2]], [price[t1], price[peak], price[t2]], 'k--', alpha=0.7)

counts = []
if len(double_top_patterns) > 0:
    counts.append(f"Double Tops: {len(double_top_patterns)}")
if len(double_bottom_patterns) > 0:
    counts.append(f"Double Bottoms: {len(double_bottom_patterns)}")

title = "Pattern Detection with Full Pattern Highlight"
if counts:
    title += " - " + ", ".join(counts)
else:
    title += " - None Detected"

ax.set_title(title)
ax.set_xlabel("Bars (Index)")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
plt.show()
