import pandas as pd


class PdUtils:
    @staticmethod
    def flatten_columns(df):
        """Flatten multi-level columns into a single level, excluding symbols."""
        if isinstance(df.columns, pd.MultiIndex):
            # Use only the first part of the column name
            df.columns = [
                col[0].strip() if isinstance(col, tuple) else col for col in df.columns
            ]
        return df
