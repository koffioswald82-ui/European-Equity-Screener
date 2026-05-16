import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
exec(open(ROOT / "app" / "pages" / "1_📊_Screener.py").read())
