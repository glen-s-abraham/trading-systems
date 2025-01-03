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


def generate_signals(data, threshold_buffer=3.0, rsi_threshold=30):
    """
    Generate Buy and Sell signals based on Bollinger Bands, Volume, and RSI filters.
    """
    df = data.copy()

    # Bollinger Band Signals
    df["Buy Signal"] = df["Adj Close"] <= (df["Bollinger_Lower"] * threshold_buffer)
    df["Sell Signal"] = df["Adj Close"] >= (
        df["Bollinger_Mid"] * (1 / threshold_buffer)
    )

    # RSI Filter for Oversold (Buy) and Overbought (Sell)
    df["Buy Signal"] = df["Buy Signal"] & (df["RSI"] < rsi_threshold)
    df["Sell Signal"] = df["Sell Signal"] & (df["RSI"] > (100 - rsi_threshold))

    # Volume Surge Filter
    df["Volume Surge"] = df["Volume"] > df["Volume"].rolling(20).mean()
    df["Buy Signal"] = df["Buy Signal"] & df["Volume Surge"]

    # Debugging signal counts
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
    Backtest a Bollinger Band-based mean reversion strategy with ATR-based stop-loss and filters.
    """
    balance = initial_balance
    position = 0  # Number of shares held
    entry_price = 0  # Price at which the position was opened
    trades = []  # To log trade details
    last_trade_date = None  # Cooldown logic

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
            last_trade_date = index

        # Sell condition
        elif position > 0:
            sell_condition = row["Sell Signal"] or row["Close"] <= atr_stop_loss

            if sell_condition:
                sell_price = row["Close"]
                trade_value = position * sell_price
                transaction_cost = trade_value * transaction_cost_rate
                net_trade_value = trade_value - transaction_cost
                profit = net_trade_value - (entry_price * position)
                balance += net_trade_value
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
                last_trade_date = index

    # Final balance calculation
    if position > 0:
        sell_price = data["Close"].iloc[-1]
        trade_value = position * sell_price
        transaction_cost = trade_value * transaction_cost_rate
        net_trade_value = trade_value - transaction_cost
        profit = net_trade_value - (entry_price * position)
        balance += net_trade_value
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

    components = IndexUtils.get_index_data("NIFTY100")
    for component in components:
        try:
            print(component)
            data = YfinanceUtils.get_historical_data(
                component, start_date=start_date, end_date=end_date
            )
            data = PdUtils.flatten_columns(data)
            data[["Bollinger_Mid", "Bollinger_Upper", "Bollinger_Lower"]] = (
                IndicatorUtils.calculate_bollinger_bands(data)
            )
            data.dropna(how="any", inplace=True)
            data["ATR"] = IndicatorUtils.calculate_atr(data)
            data.dropna(how="any", inplace=True)
            data["RSI"] = IndicatorUtils.calculate_rsi(data)
            data.dropna(how="any", inplace=True)
            data = generate_signals(data, threshold_buffer=3.0)

            summary, trades = backtest(data)
            symbol_performance_matrix[component] = summary
            print("------------------------------------------")
        except Exception as e:
            print(f"Error processing {component}: {e}")
            continue

    # Save results to JSON
    with open("./optimized_mean_reversion_bb_test.json", "w") as json_file:
        json.dump(symbol_performance_matrix, json_file, indent=4)


if __name__ == "__main__":
    main()
