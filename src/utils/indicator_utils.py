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
    def calculate_atr(data, n=14):
        df = data.copy()
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
        df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
        df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()

        return df["ATR"]
