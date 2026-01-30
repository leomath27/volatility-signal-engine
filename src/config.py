from pathlib import Path

# Get project root by going up from src/config.py
# This works when config.py is in src/ directory
_file_path = Path(__file__).resolve()
if _file_path.name == "config.py" and _file_path.parent.name == "src":
    PROJECT_ROOT = _file_path.parents[1]
else:
    # Fallback: search for requirements.txt or other project markers
    current = _file_path.parent
    while current != current.parent:
        if (current / "requirements.txt").exists():
            PROJECT_ROOT = current
            break
        current = current.parent
    else:
        PROJECT_ROOT = _file_path.parents[1]  # Default fallback

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SYMBOL = "^GSPC"   # S&P 500 index (best)
START_DATE = "2015-01-01"
END_DATE = None

RANDOM_STATE = 42
