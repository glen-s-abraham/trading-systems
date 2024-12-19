import sys
import os
import json
import pandas as pd
import numpy as np

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import IndexUtils
from utils import YfinanceUtils
from utils import IndicatorUtils
from utils import PdUtils

def generate_orb_signals(data, opening_range_minutes=30):
    """
    Generate signals based on the Opening Range Breakout (ORB) strategy using intraday data.
    We assume that 'High', 'Low', and 'Close' columns exist, and that the DataFrame index is a DatetimeIndex.
    """
    df = data.copy()

    # Ensure we have a proper DatetimeIndex
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a DatetimeIndex. Please ensure the data has a proper Datetime index.")

    # Extract the date part for grouping (one group per day)
    df["Date"] = df.index.date

    def find_opening_range(group):
        # day_start should be the first timestamp of that day
        day_start = group.index[0]
        cutoff = day_start + pd.Timedelta(minutes=opening_range_minutes)
        morning_data = group[group.index <= cutoff]

        if len(morning_data) > 0:
            opening_high = morning_data["High"].max()
            opening_low = morning_data["Low"].min()
        else:
            opening_high = np.nan
            opening_low = np.nan

        group["Opening_High"] = opening_high
        group["Opening_Low"] = opening_low
        return group

    # Apply find_opening_range to each day
    df = df.groupby("Date", group_keys=False).apply(find_opening_range)

    # Generate Buy/Sell signals:  
    # Buy if Close > Opening_High and Sell if Close < Opening_Low
    df["Buy Signal"] = (df["Close"] > df["Opening_High"]) & df["Opening_High"].notna()
    df["Sell Signal"] = (df["Close"] < df["Opening_Low"]) & df["Opening_Low"].notna()

    print("Buy Signals:", df["Buy Signal"].sum())
    print("Sell Signals:", df["Sell Signal"].sum())

    return df


def backtest(
    data,
    initial_balance=100000,
    transaction_cost_rate=0.001,
    risk_pct=0.025,
    atr_multiplier=2.0,
):
    """
    Backtest the ORB strategy with ATR-based stop-loss.

    Parameters:
        data: DataFrame with 'Close', 'Buy Signal', 'Sell Signal', and 'ATR' columns.
        initial_balance: Starting capital.
        transaction_cost_rate: Fraction of trade value taken as cost.
        risk_pct: Fraction of capital risked per trade.
        atr_multiplier: Multiplier for ATR-based stop calculation.
    """
    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []
    last_trade_date = None

    for index, row in data.iterrows():
        # Use ATR if available, else fallback
        if "ATR" in data.columns:
            row_atr = row["ATR"]
        else:
            row_atr = 1.0  # Dummy ATR if not calculated

        # Stop-loss calculation for a long position
        if position > 0:
            atr_stop_loss = entry_price - (atr_multiplier * row_atr)
        else:
            atr_stop_loss = None

        # Entry Condition: Buy if flat and Buy Signal is True
        if row.get("Buy Signal", False) and position == 0:
            if atr_stop_loss is not None:
                stop_loss_distance = row["Close"] - atr_stop_loss
            else:
                # If no ATR, assume a fixed fractional stop
                stop_loss_distance = row["Close"] * 0.01

            # Calculate position size based on risk
            risk_per_trade = balance * risk_pct
            position_size = risk_per_trade / stop_loss_distance

            # Adjust for transaction costs on entry
            position = position_size * (1 - transaction_cost_rate)
            entry_price = row["Close"]
            balance -= position * entry_price * (1 + transaction_cost_rate)
            trades.append({"Action": "BUY", "Index": index, "Price": entry_price, "Quantity": position, "Balance": balance})
            print(f"BUY: {index} - Price: {entry_price:.2f}, Quantity: {position:.2f}")
            last_trade_date = index

        # Exit Condition: If in a position, sell if Sell Signal is True or Stop is hit
        elif position > 0:
            sell_condition = row.get("Sell Signal", False) or (atr_stop_loss is not None and row["Close"] <= atr_stop_loss)

            if sell_condition:
                sell_price = row["Close"]
                trade_value = position * sell_price
                transaction_cost = trade_value * transaction_cost_rate
                net_trade_value = trade_value - transaction_cost
                profit = net_trade_value - (entry_price * position)
                balance += net_trade_value
                trades.append({"Action": "SELL", "Index": index, "Price": sell_price, "Profit": profit, "Balance": balance})
                print(f"SELL: {index} - Price: {sell_price:.2f}, Profit: {profit:.2f}")
                position = 0.0
                entry_price = 0.0
                last_trade_date = index

    # If still holding a position at the end of the period, close it
    if position > 0:
        sell_price = data["Close"].iloc[-1]
        trade_value = position * sell_price
        transaction_cost = trade_value * transaction_cost_rate
        net_trade_value = trade_value - transaction_cost
        profit = net_trade_value - (entry_price * position)
        balance += net_trade_value
        trades.append({"Action": "FINAL SELL", "Index": data.index[-1], "Price": sell_price, "Profit": profit, "Balance": balance})
        print(f"FINAL SELL: Last Price: {sell_price:.2f}, Final Profit: {profit:.2f}")

    # Compile trade results
    trades_df = pd.DataFrame(trades)

    # Compute performance metrics
    total_trades = len(trades_df[trades_df["Action"].isin(["SELL", "FINAL SELL"])])
    winning_trades = trades_df[(trades_df["Action"].isin(["SELL", "FINAL SELL"])) & (trades_df["Profit"] > 0)].shape[0]
    losing_trades = trades_df[(trades_df["Action"].isin(["SELL", "FINAL SELL"])) & (trades_df["Profit"] < 0)].shape[0]
    max_profit = trades_df["Profit"].max() if "Profit" in trades_df and not trades_df["Profit"].isna().all() else 0
    max_loss = trades_df["Profit"].min() if "Profit" in trades_df and not trades_df["Profit"].isna().all() else 0
    total_profit = trades_df["Profit"].sum() if "Profit" in trades_df else 0

    summary = {
        "Final Balance": balance,
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Max Profit": max_profit,
        "Max Loss": max_loss,
        "Total Profit": total_profit,
    }

    return summary, trades_df


def main():
    start_date = "2024-11-18"
    end_date = "2024-12-18"
    interval = "15m"  # Intraday interval for ORB
    symbol_performance_matrix = {}

    # Retrieve the list of components from NIFTY100
    components = IndexUtils.get_index_data("NIFTY100")

    for component in components:
        try:
            print(f"Processing {component}...")
            data = YfinanceUtils.get_historical_data(component, start_date=start_date, end_date=end_date, interval=interval)

            # Check if data is returned
            if data is None or data.empty:
                print(f"No data for {component} in the given period.")
                continue

            data = PdUtils.flatten_columns(data)

            # Ensure index is DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Calculate ATR
            data["ATR"] = IndicatorUtils.calculate_atr(data)
            data.dropna(how="any", inplace=True)  # Drop rows with missing values

            # Generate ORB signals
            data = generate_orb_signals(data, opening_range_minutes=30)

            # Only backtest if data is still valid and not empty after signal generation
            if data.empty:
                print(f"No valid data after signal generation for {component}.")
                continue

            # Backtest the ORB strategy
            summary, trades = backtest(data)
            symbol_performance_matrix[component] = summary
            print(f"Completed {component}")
            print("------------------------------------------")

        except Exception as e:
            print(f"Error processing {component}: {e}")
            continue

    # Save the performance results to JSON
    output_path = "./orb_strategy_results.json"
    with open(output_path, "w") as json_file:
        json.dump(symbol_performance_matrix, json_file, indent=4)

    print("Backtesting complete. Results saved to orb_strategy_results.json.")


if __name__ == "__main__":
    main()
