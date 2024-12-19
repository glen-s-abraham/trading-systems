import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to the system path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import YfinanceUtils
from utils import PdUtils

#########################################
# Load Historical Data
#########################################
# Try a longer historical range to increase chances of finding a pattern
historic_data = YfinanceUtils.get_historical_data("RELIANCE.NS", "2019-12-19", "2024-12-19")
df = PdUtils.flatten_columns(historic_data)

#########################################
# Parameters
#########################################
# A looser tolerance is no longer strictly necessary since we are using ranges,
# but we keep it just in case we do a ratio check.
fibo_tolerance = 0.05  
pivot_lookback = 3  # Increasing lookback to identify stronger pivots

#########################################
# Helper Functions
#########################################
def is_pivot_high(df, i, lb):
    """Check if df.High[i] is a pivot high."""
    if i-lb < 0 or i+lb >= len(df):
        return False
    return df["High"].iloc[i] == max(df["High"].iloc[i - lb:i + lb + 1])

def is_pivot_low(df, i, lb):
    """Check if df.Low[i] is a pivot low."""
    if i-lb < 0 or i+lb >= len(df):
        return False
    return df["Low"].iloc[i] == min(df["Low"].iloc[i - lb:i + lb + 1])

def detect_pivots(df, lb=2):
    pivots = []
    for i in range(len(df)):
        if is_pivot_high(df, i, lb):
            pivots.append((i, df["High"].iloc[i], "high"))
        elif is_pivot_low(df, i, lb):
            pivots.append((i, df["Low"].iloc[i], "low"))
    return pivots

def is_in_range(value, min_val, max_val):
    return min_val <= value <= max_val

#########################################
# Elliott Wave Identification
#########################################
def identify_12345_pattern(pivots):
    """
    Attempts to identify a 1-2-3-4-5 Elliott wave pattern in the given pivots.
    Using more relaxed ratio conditions:
    - Pattern: low-high-low-high-low for an uptrend
    - Wave 2 retracement: 38% to 78%
    - Wave 3 extension: >100% of wave 1 length (just longer than wave 1)
    - Wave 4 retracement: 20% to 62%
    """
    if len(pivots) < 5:
        return None

    candidates = []
    for a in range(len(pivots)-4):
        p1 = pivots[a]    # (index, price, type)
        p2 = pivots[a+1]
        p3 = pivots[a+2]
        p4 = pivots[a+3]
        p5 = pivots[a+4]

        # Check pattern of types: low-high-low-high-low
        if p1[2] == "low" and p2[2] == "high" and p3[2] == "low" and p4[2] == "high" and p5[2] == "low":
            wave1_len = p2[1] - p1[1]
            wave2_retrace = (p3[1] - p2[1]) / wave1_len if wave1_len != 0 else np.nan
            wave3_ext = (p4[1] - p3[1]) / abs(wave1_len) if wave1_len != 0 else np.nan
            wave4_retrace = (p5[1] - p4[1]) / (p4[1] - p3[1]) if (p4[1] - p3[1]) != 0 else np.nan

            # Relaxed Conditions
            # Wave 2: between 0.38 and 0.78 retracement (wave2_retrace is negative, so we use -wave2_retrace)
            w2_ok = is_in_range(-wave2_retrace, 0.38, 0.78)
            # Wave 3: simply greater than 1.0 (longer than wave 1)
            w3_ok = (wave3_ext > 1.0)
            # Wave 4: between 0.2 and 0.62 retracement
            w4_ok = is_in_range(-wave4_retrace, 0.2, 0.62)

            if w2_ok and w3_ok and w4_ok:
                candidates.append((p1, p2, p3, p4, p5))

            # Debug prints to understand why candidates fail
            else:
                print(f"Candidate at pivots {a}-{a+4} rejected:")
                print(f"p1={p1}, p2={p2}, p3={p3}, p4={p4}, p5={p5}")
                print(f"wave1_len={wave1_len}, wave2_retrace={wave2_retrace}, wave3_ext={wave3_ext}, wave4_retrace={wave4_retrace}")
                print(f"w2_ok={w2_ok}, w3_ok={w3_ok}, w4_ok={w4_ok}\n")

    if candidates:
        return candidates[-1]
    else:
        return None

def identify_abc_pattern(pivots, wave5_pivot):
    """
    Attempts to identify an A-B-C corrective pattern after wave 5 pivot.
    Pattern: high-low-high for a simple zigzag correction.
    We won't be too strict on ratios here. We'll just ensure it looks like a correction.
    Relaxed approach: 
    - B retracement of A: between ~30% and 70%
    - C similar or up to 1.7 times A
    """
    start_index = wave5_pivot[0]
    future_pivots = [p for p in pivots if p[0] > start_index]

    if len(future_pivots) < 3:
        return None

    for i in range(len(future_pivots)-2):
        A = future_pivots[i]
        B = future_pivots[i+1]
        C = future_pivots[i+2]

        if A[2] == "high" and B[2] == "low" and C[2] == "high":
            a_len = A[1] - wave5_pivot[1]
            b_retrace = (B[1] - A[1]) / a_len if a_len != 0 else np.nan
            c_len = C[1] - B[1]
            c_ratio = c_len / abs(a_len) if a_len != 0 else np.nan

            # Relaxed conditions for A-B-C:
            # B retrace between 30% and 70%
            b_ok = is_in_range(-b_retrace, 0.3, 0.7)
            # C ~ A or a bit more: up to 1.7 times
            c_ok = (c_ratio > 0.5 and c_ratio < 1.7)

            if b_ok and c_ok:
                return (A, B, C)

    return None

#########################################
# Main Logic
#########################################
pivots = detect_pivots(df, lb=pivot_lookback)
pattern = identify_12345_pattern(pivots)

if pattern is None:
    print("No 1-5 Elliott Wave pattern found with relaxed conditions.")
else:
    print("1-5 Elliott Wave pattern detected with relaxed conditions.")

abc_pattern = None
if pattern:
    p1, p2, p3, p4, p5 = pattern
    abc_pattern = identify_abc_pattern(pivots, p5)
    if abc_pattern is None:
        print("No A-B-C corrective pattern found following the 1-5 pattern.")
    else:
        print("A-B-C corrective pattern detected with relaxed conditions.")

#########################################
# Visualization
#########################################
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price", color='black')

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Price", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

pivot_x = [df.index[p[0]] for p in pivots]
pivot_y = [p[1] for p in pivots]
ax.scatter(pivot_x, pivot_y, c='blue', s=50, zorder=5, label="Pivots")

if pattern:
    wave_pivots = [p1, p2, p3, p4, p5]
    wave_labels = ["W1", "W2", "W3", "W4", "W5"]
    wave_x = [df.index[p[0]] for p in wave_pivots]
    wave_y = [p[1] for p in wave_pivots]
    ax.plot(wave_x, wave_y, color='green', linestyle='-', linewidth=2, marker='o', markersize=8, label="1-5 Waves")

    for lbl, wave_pivot in zip(wave_labels, wave_pivots):
        ax.text(df.index[wave_pivot[0]], wave_pivot[1]*1.001, lbl, color='green', fontsize=12, 
                ha='left', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.8))

if abc_pattern:
    A, B, C = abc_pattern
    abc_pivots = [A, B, C]
    abc_labels = ["A", "B", "C"]
    abc_x = [df.index[p[0]] for p in abc_pivots]
    abc_y = [p[1] for p in abc_pivots]
    ax.plot(abc_x, abc_y, color='red', linestyle='--', linewidth=2, marker='o', markersize=8, label="A-B-C Waves")

    for lbl, abc_pivot in zip(abc_labels, abc_pivots):
        ax.text(df.index[abc_pivot[0]], abc_pivot[1]*1.001, lbl, color='red', fontsize=12, 
                ha='left', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.8))

ax.set_title("Elliott Wave Detection (Relaxed Conditions)", fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()
