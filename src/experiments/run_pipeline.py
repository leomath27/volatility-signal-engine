from __future__ import annotations

import json

from src.config import SYMBOL
from src.models.train_xgb import DEFAULT_HORIZONS, train_many


if __name__ == "__main__":
    summary = train_many(symbol=SYMBOL, horizons=DEFAULT_HORIZONS, refresh_data=True)
    print(json.dumps(summary, indent=2))
