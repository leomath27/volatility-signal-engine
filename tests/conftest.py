import os
import tempfile

# Must be set before any `src` import: isolated artifact dir + offline synthetic market data
os.environ.setdefault("VOL_HOME", tempfile.mkdtemp(prefix="volsig_"))
os.environ["VOL_DATA_SOURCE"] = "synthetic"
