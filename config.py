from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE / "data" / "uploads"
DB_PATH = BASE / "data" / "neuro.db"
ALLOWED_EXT = {".csv", ".xlsx", ".xls", ".json", ".txt", ".pdf"}
MAX_FILE_MB = 25
MAX_ROWS = 500_000
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR = BASE / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = BASE / "data" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RATE_LIMIT_PER_MIN = 60
