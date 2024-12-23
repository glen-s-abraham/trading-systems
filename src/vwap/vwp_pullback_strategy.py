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


def generate_vwap_pullback_signals(data):
    df = data.copy()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a DatetimeIndex.")

    # Calculate VWAP
    df["VWAP"] = IndicatorUtils.calculate_vwap(df)

    # Identify trend direction (above or below VWAP)
    df["Trend"] = np.where(df["Close"] > df["VWAP"], "Uptrend", "Downtrend")

    # Pullback signals
    df["Buy Signal"] = (
        (df["Trend"] == "Uptrend")
        & (df["Low"] <= df["VWAP"])
        & (df["Close"] > df["VWAP"])  # Price closes above VWAP after touching it
    )
    df["Sell Signal"] = (
        (df["Trend"] == "Downtrend")
        & (df["High"] >= df["VWAP"])
        & (df["Close"] < df["VWAP"])  # Price closes below VWAP after touching it
    )

    print("Buy Signals:", df["Buy Signal"].sum())
    print("Sell Signals:", df["Sell Signal"].sum())

    return df


def backtest(
    data,
    initial_balance=100000,
    risk_pct=0.005,  # Reduced risk per trade
    atr_multiplier=1.5,  # Reduced stop-loss distance
    profit_target=2.0,  # Profit target adjusted
    trailing_stop_factor=1.0,  # Trailing stop for capturing trends
):
    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []

    for index, row in data.iterrows():
        row_atr = row.get("ATR", 1.0)

        # Entry condition
        if row.get("Buy Signal", False) and position == 0:
            atr_stop_loss = row["Close"] - (atr_multiplier * row_atr)
            stop_loss_distance = row["Close"] - atr_stop_loss

            risk_per_trade = balance * risk_pct
            position_size = risk_per_trade / stop_loss_distance

            trade_value = position_size * row["Close"]
            percentage_cost = trade_value * 0.0025
            transaction_cost = min(20, percentage_cost)

            position = position_size
            entry_price = row["Close"]
            balance -= trade_value + transaction_cost

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

        # Exit condition
        elif position > 0:
            initial_risk = entry_price - (entry_price - stop_loss_distance)
            profit_target_price = entry_price + profit_target * initial_risk

            stop_hit = row["Close"] <= atr_stop_loss
            profit_target_hit = row["Close"] >= profit_target_price
            sell_signal = row.get("Sell Signal", False)

            if stop_hit or profit_target_hit or sell_signal:
                exit_price = row["Close"]
                trade_value = position * exit_price

                percentage_cost = trade_value * 0.0025
                transaction_cost = min(20, percentage_cost)

                net_trade_value = trade_value - transaction_cost
                profit = net_trade_value - (entry_price * position)
                balance += net_trade_value

                trades.append(
                    {
                        "Action": "SELL",
                        "Index": index,
                        "Price": exit_price,
                        "Profit": profit,
                        "Balance": balance,
                    }
                )

                print(f"SELL: {index} - Price: {exit_price:.2f}, Profit: {profit:.2f}")

                position = 0.0
                entry_price = 0.0

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df[trades_df["Action"] == "SELL"])
    winning_trades = trades_df[trades_df["Profit"] > 0].shape[0]
    losing_trades = trades_df[trades_df["Profit"] < 0].shape[0]
    max_profit = trades_df["Profit"].max()
    max_loss = trades_df["Profit"].min()
    total_profit = trades_df["Profit"].sum()

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
    start_date = "2024-11-01"
    end_date = "2024-12-23"
    interval = "15m"
    symbol_performance_matrix = {}

    components = IndexUtils.get_index_data("NIFTY100")
    for component in components:
        try:
            print(f"Processing {component}...")
            data = YfinanceUtils.get_historical_data(
                component, start_date=start_date, end_date=end_date, interval=interval
            )

            if data is None or data.empty:
                print(f"No data for {component} in the given period.")
                continue

            data = PdUtils.flatten_columns(data)

            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            data["ATR"] = IndicatorUtils.calculate_atr(data)
            data.dropna(how="any", inplace=True)

            data = generate_vwap_pullback_signals(data)

            if data.empty:
                print(f"No valid data after signal generation for {component}.")
                continue

            summary, trades = backtest(
                data,
                initial_balance=15000,
                risk_pct=0.005,
                atr_multiplier=1.5,
                profit_target=2.0,
                trailing_stop_factor=1.0,
            )

            symbol_performance_matrix[component] = summary
            print(f"Completed {component}")
            print("------------------------------------------")

        except Exception as e:
            print(f"Error processing {component}: {e}")
            continue

    with open("./vwap_pullback_strategy_results.json", "w") as json_file:
        json.dump(symbol_performance_matrix, json_file, indent=4)

    print(
        "VWAP Pullback backtesting complete. Results saved to vwap_pullback_strategy_results.json."
    )


if __name__ == "__main__":
    main()
