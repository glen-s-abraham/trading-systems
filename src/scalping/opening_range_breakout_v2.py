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
    df = data.copy()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a DatetimeIndex.")

    # Create a 'Date' column from the index
    df["Date"] = df.index.date

    def find_opening_range(group):
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

    # Exclude the "Date" column from apply operation
    cols = [c for c in df.columns if c != "Date"]
    df = df.groupby("Date", group_keys=False)[cols].apply(find_opening_range)

    # Generate basic Buy/Sell signals
    df["Buy Signal"] = (df["Close"] > df["Opening_High"]) & df["Opening_High"].notna()
    df["Sell Signal"] = (df["Close"] < df["Opening_Low"]) & df["Opening_Low"].notna()

    print("Buy Signals:", df["Buy Signal"].sum())
    print("Sell Signals:", df["Sell Signal"].sum())

    return df


def backtest(
    data,
    initial_balance=100000,
    risk_pct=0.01,  # Tighter risk per trade
    atr_multiplier=2.0,
    profit_target=2.0,  # Take profit at 2R
    trailing_stop_factor=1.0,  # Trailing stop at 1 * ATR once in profit
):
    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []
    last_trade_date = None

    initial_stop_loss = None
    use_trailing_stop = False

    for index, row in data.iterrows():
        row_atr = row.get("ATR", 1.0)

        # If in a position, determine current stop
        if position > 0:
            if use_trailing_stop:
                current_stop = max(
                    initial_stop_loss, row["Close"] - (trailing_stop_factor * row_atr)
                )
            else:
                current_stop = initial_stop_loss
        else:
            current_stop = None

        # Entry (Buy) condition
        if row.get("Buy Signal", False) and position == 0:
            atr_stop_loss = row["Close"] - (atr_multiplier * row_atr)
            stop_loss_distance = row["Close"] - atr_stop_loss

            risk_per_trade = balance * risk_pct
            position_size = risk_per_trade / stop_loss_distance

            # Calculate trade value
            trade_value = position_size * row["Close"]
            # Transaction cost: min(20, 0.25% of trade value)
            percentage_cost = trade_value * 0.0025
            transaction_cost = min(20, percentage_cost)

            # Enter position
            position = position_size
            entry_price = row["Close"]
            # Deduct trade_value + transaction_cost from balance
            balance -= trade_value + transaction_cost

            initial_stop_loss = atr_stop_loss
            use_trailing_stop = False

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

        # Exit conditions if in a position
        elif position > 0:
            initial_risk = entry_price - initial_stop_loss
            profit_target_price = entry_price + profit_target * initial_risk

            # Activate trailing stop if reached at least 1R profit
            if row["Close"] >= (entry_price + initial_risk):
                use_trailing_stop = True

            stop_hit = current_stop is not None and row["Close"] <= current_stop
            profit_target_hit = row["Close"] >= profit_target_price
            sell_signal = row.get("Sell Signal", False)

            if stop_hit or profit_target_hit or sell_signal:
                exit_price = row["Close"]
                trade_value = position * exit_price

                # Transaction cost on sell
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
                initial_stop_loss = None
                use_trailing_stop = False
                last_trade_date = index

    # Close any open position at the end
    if position > 0:
        sell_price = data["Close"].iloc[-1]
        trade_value = position * sell_price

        # Transaction cost on final sell
        percentage_cost = trade_value * 0.0025
        transaction_cost = min(20, percentage_cost)

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

    # Performance metrics
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df[trades_df["Action"].isin(["SELL", "FINAL SELL"])])
    winning_trades = trades_df[
        (trades_df["Action"].isin(["SELL", "FINAL SELL"])) & (trades_df["Profit"] > 0)
    ].shape[0]
    losing_trades = trades_df[
        (trades_df["Action"].isin(["SELL", "FINAL SELL"])) & (trades_df["Profit"] < 0)
    ].shape[0]
    max_profit = (
        trades_df["Profit"].max()
        if "Profit" in trades_df and not trades_df["Profit"].isna().all()
        else 0
    )
    max_loss = (
        trades_df["Profit"].min()
        if "Profit" in trades_df and not trades_df["Profit"].isna().all()
        else 0
    )
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
    start_date = "2024-09-01"
    end_date = "2024-09-30"
    interval = "15m"
    symbol_performance_matrix = {}

    # components = IndexUtils.get_index_data("NIFTY100")
    components = ["MARICO.NS", "BHARTIARTL.NS", "TITAN.NS", "SHREECEM.NS", "LT.NS"]
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

            # Call generate_orb_signals with only the parameters it supports
            data = generate_orb_signals(data, opening_range_minutes=30)

            if data.empty:
                print(f"No valid data after signal generation for {component}.")
                continue

            summary, trades = backtest(
                data,
                initial_balance=10000,
                risk_pct=0.01,
                atr_multiplier=2.0,
                profit_target=2.0,
                trailing_stop_factor=1.0,
            )

            symbol_performance_matrix[component] = summary
            print(f"Completed {component}")
            print("------------------------------------------")

        except Exception as e:
            print(f"Error processing {component}: {e}")
            continue

    with open("./orb_strategy_optimized_results.json", "w") as json_file:
        json.dump(symbol_performance_matrix, json_file, indent=4)

    print(
        "Optimized backtesting complete. Results saved to orb_strategy_optimized_results.json."
    )


if __name__ == "__main__":
    main()
