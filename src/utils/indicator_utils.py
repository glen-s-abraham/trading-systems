import pandas as pd


class IndicatorUtils:
    @staticmethod
    def calculate_z_score(data, lookback=20):
        df = data.copy()
        df["Rolling Mean"] = data["Adj Close"].rolling(window=lookback).mean()
        df["Rolling Std"] = data["Adj Close"].rolling(window=lookback).std()
        df["Z-Score"] = (df["Adj Close"] - df["Rolling Mean"]) / df["Rolling Std"]
        return df["Z-Score"]

    @staticmethod
    def calculate_atr(data, period=14):
        df = data.copy()
        # Calculate True Range (TR)
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
        df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)

        # Apply Wilder's Smoothing using Exponential Moving Average
        df["ATR"] = df["TR"].ewm(alpha=1 / period, adjust=False).mean()

        return df["ATR"]

    @staticmethod
    def calculate_rsi(data, period=14):
        df = data.copy()
        df["DELTA"] = df["Adj Close"].diff(1)  # Price differences

        # Separate gains and losses
        df["GAIN"] = df["DELTA"].apply(lambda x: x if x > 0 else 0)
        df["LOSS"] = -df["DELTA"].apply(lambda x: x if x < 0 else 0)

        # Calculate rolling averages of gains and losses
        df["AVG_GAIN"] = df["GAIN"].ewm(alpha=1 / period, adjust=False).mean()
        df["AVG_LOSS"] = df["LOSS"].ewm(alpha=1 / period, adjust=False).mean()
        # Calculate Relative Strength (RS)
        df["RS"] = df["AVG_GAIN"] / df["AVG_LOSS"]

        # Calculate RSI
        df["RSI"] = 100 - (100 / (1 + df["RS"]))

        return df["RSI"]

    @staticmethod
    def calculate_bollinger_bands(data, lookback=20):
        df = data.copy()
        if "Adj Close" not in df.columns:
            raise ValueError("The input DataFrame must contain an 'Adj Close' column.")

        # Rolling Mean and Standard Deviation
        df["Bollinger_Mid"] = (
            df["Adj Close"].rolling(window=lookback, min_periods=lookback).mean()
        )
        df["Rolling Std"] = (
            df["Adj Close"].rolling(window=lookback, min_periods=lookback).std()
        )

        # Bollinger Bands
        df["Bollinger_Upper"] = df["Bollinger_Mid"] + (2 * df["Rolling Std"])
        df["Bollinger_Lower"] = df["Bollinger_Mid"] - (2 * df["Rolling Std"])

        return df[["Bollinger_Mid", "Bollinger_Upper", "Bollinger_Lower"]]

    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        df = data.copy()
        if "Adj Close" not in df.columns:
            raise ValueError("The input DataFrame must contain an 'Adj Close' column.")

        # Calculate Fast and Slow EMA
        df["Fast_EMA"] = df["Adj Close"].ewm(span=fast_period, adjust=False).mean()
        df["Slow_EMA"] = df["Adj Close"].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD Line and Signal Line
        df["MACD_Line"] = df["Fast_EMA"] - df["Slow_EMA"]
        df["Signal_Line"] = df["MACD_Line"].ewm(span=signal_period, adjust=False).mean()

        # Calculate MACD Histogram
        df["MACD_Histogram"] = df["MACD_Line"] - df["Signal_Line"]

        return df[["MACD_Line", "Signal_Line", "MACD_Histogram"]]

    @staticmethod
    def calculate_vwap(data):
        df = data.copy()
        # Calculate Typical Price
        df["Typical Price"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3

        # Calculate Cumulative Values
        df["Cumulative TPV"] = (df["Typical Price"] * df["Volume"]).cumsum()
        df["Cumulative Volume"] = df["Volume"].cumsum()

        # Calculate VWAP
        df["VWAP"] = df["Cumulative TPV"] / df["Cumulative Volume"]

        return df["VWAP"]
