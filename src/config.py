from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SYMBOL = "^GSPC"   # S&P 500 index (best)
START_DATE = "2015-01-01"
END_DATE = None

RANDOM_STATE = 42
