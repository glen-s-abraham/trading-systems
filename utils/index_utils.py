import json
import os
from typing import Literal, Dict

# Define allowed keys
INDEX_KEYS = Literal["NIFTY100", "NIFTY_MIDCAP", "NIFTY_SMALLCAP"]

# Map index keys to their corresponding JSON keys
INDEX_MAPPING = {
    "NIFTY100": "nifty_100",
    "NIFTY_MIDCAP": "nifty_midcap",
    "NIFTY_SMALLCAP": "nifty_smallcap",
}

class IndexUtils:
    def __init__(self, file_path: str = None):
        # Default file path to utils/data/indices.json
        if file_path is None:
            self.file_path = os.path.join(
                os.path.dirname(__file__), "data", "indices.json"
            )
        else:
            self.file_path = file_path

        self.data = self._load_index_data()

    def _load_index_data(self) -> Dict:
        try:
            with open(self.file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.file_path}' not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from '{self.file_path}': {e}")

    def get_index_data(self, key: INDEX_KEYS):
        if key not in INDEX_MAPPING:
            raise ValueError(
                f"Invalid key '{key}'. Valid keys are: {list(INDEX_MAPPING.keys())}"
            )
        return self.data[INDEX_MAPPING[key]]
