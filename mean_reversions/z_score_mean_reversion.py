import sys
import os

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np
from utils import IndexUtils, INDEX_KEYS


def get_index_components(index):
    import yfinance as yf

    idx = yf.Ticker(index)
    components = idx.get_index_members()
    # Display the list of NIFTY 50 stocks
    return list(components.keys())


def get_historical_data(symbol, start_date, end_date, interval="1d"):
    import yfinance as yf
    import pandas as pd

    # Fetch OHLCV data from Yahoo Finance using yfinance library
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if data.empty:
            raise ValueError(
                f"No data found for symbol {symbol} between {start_date} and {end_date}."
            )

        # Ensure the DataFrame is properly formatted
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)  # Reset index to include date as a column
        df.dropna(how="any", inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def flatten_columns(df):
    """Flatten multi-level columns into a single level, excluding symbols."""
    if isinstance(df.columns, pd.MultiIndex):
        # Use only the first part of the column name
        df.columns = [
            col[0].strip() if isinstance(col, tuple) else col for col in df.columns
        ]
    return df


def calculate_z_score(data, lookback=20):
    df = data.copy()
    # Calculate Rolling Mean and Std
    df["Rolling Mean"] = data["Adj Close"].rolling(window=lookback).mean()
    df["Rolling Std"] = data["Adj Close"].rolling(window=lookback).std()

    df["Z-Score"] = (df["Adj Close"] - df["Rolling Mean"]) / df["Rolling Std"]
    df.dropna(how="any", inplace=True)

    return df


def calculate_atr(DF, n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    df.dropna(how="any", inplace=True)
    return df


def generate_signals(data, z_threshold=2.5):
    df = data.copy()
    df["Buy Signal"] = data["Z-Score"] < -z_threshold  # Oversold
    df["Sell Signal"] = data["Z-Score"] > z_threshold  # Overbought
    return df


import pandas as pd


def backtest(
    data,
    initial_balance=100000,
    transaction_cost_rate=0.001,
    risk_pct=0.025,
    atr_multiplier=1.5,
):
    balance = initial_balance
    position = 0  # Number of shares held
    entry_price = 0  # Price at which the position was opened
    trades = []  # To log trade details

    for index, row in data.iterrows():
        # Calculate ATR-based stop-loss dynamically
        atr_stop_loss = row["Close"] - (atr_multiplier * row["ATR"])

        # Buy condition
        if row["Buy Signal"] and position == 0:
            # Determine position size based on risk
            risk_per_trade = balance * risk_pct
            stop_loss_distance = row["Close"] - atr_stop_loss
            position_size = risk_per_trade / stop_loss_distance

            # Adjust for transaction costs
            position = position_size * (1 - transaction_cost_rate)
            entry_price = row["Close"]
            balance -= position * entry_price * (1 + transaction_cost_rate)
            trades.append(
                {
                    "Action": "BUY",
                    "Index": index,
                    "Price": entry_price,
                    "Quantity": position,
                    "Balance": balance,
                }
            )
            print(f"BUY: {index} - Price: {entry_price:.2f}, Quantity: {position:.2f}")

        # Sell condition
        elif position > 0:
            sell_condition = row["Sell Signal"] or row["Close"] <= atr_stop_loss

            if sell_condition:
                sell_price = row["Close"]
                trade_value = position * sell_price
                transaction_cost = trade_value * transaction_cost_rate
                balance += trade_value - transaction_cost
                profit = trade_value - (entry_price * position)
                trades.append(
                    {
                        "Action": "SELL",
                        "Index": index,
                        "Price": sell_price,
                        "Profit": profit,
                        "Balance": balance,
                    }
                )
                print(f"SELL: {index} - Price: {sell_price:.2f}, Profit: {profit:.2f}")
                position = 0  # Clear position
                entry_price = 0  # Reset entry price

    # Final balance calculation
    if position > 0:
        sell_price = data["Close"].iloc[-1]
        trade_value = position * sell_price
        transaction_cost = trade_value * transaction_cost_rate
        balance += trade_value - transaction_cost
        profit = trade_value - (entry_price * position)
        trades.append(
            {
                "Action": "FINAL SELL",
                "Index": data.index[-1],
                "Price": sell_price,
                "Profit": profit,
                "Balance": balance,
            }
        )
        print(f"FINAL SELL: Last Price: {sell_price:.2f}, Final Profit: {profit:.2f}")

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)

    # Calculate trade statistics
    total_trades = len(trades_df[trades_df["Action"] == "SELL"])
    winning_trades = trades_df[
        (trades_df["Action"] == "SELL") & (trades_df["Profit"] > 0)
    ].shape[0]
    losing_trades = trades_df[
        (trades_df["Action"] == "SELL") & (trades_df["Profit"] < 0)
    ].shape[0]
    max_profit = trades_df["Profit"].max() if "Profit" in trades_df else 0
    max_loss = trades_df["Profit"].min() if "Profit" in trades_df else 0
    total_profit = trades_df["Profit"].sum()

    # Summary dictionary
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
    start_date = "2019-12-18"
    end_date = "2024-12-18"
    symbol_performance_matrix = {}
    index_utils = IndexUtils()
    components = index_utils.get_index_data("NIFTY100")
    for component in components:
        try:
            print(component)
            data = get_historical_data(
                component, start_date=start_date, end_date=end_date
            )
            data = flatten_columns(data)
            data = calculate_z_score(data)
            data = calculate_atr(data)
            data = generate_signals(data, 2.5)
            balance, trades = backtest(data)
            symbol_performance_matrix[component] = balance
            print("------------------------------------------")
        except:
            continue

    with open("./mean_reversion_z_score_test.json", "w") as json_file:
        json.dump(symbol_performance_matrix, json_file, indent=4)



main()
