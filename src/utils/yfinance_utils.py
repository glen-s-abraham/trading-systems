import yfinance as yf
import pandas as pd


class YfinanceUtils:
    
    @staticmethod
    def get_historical_data(symbol, start_date, end_date, interval="1d"):


        # Fetch OHLCV data from Yahoo Finance using yfinance library
        try:
            data = yf.download(
                symbol, start=start_date, end=end_date, interval=interval,timeout=1
            )
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
