"""Safe file loading. Files are only parsed as data, never executed.
Originals are stored untouched; all cleaning works on copies."""
import uuid, json, io
from pathlib import Path
import pandas as pd
from .. import config

class LoadError(Exception):
    pass

def save_upload(file_storage):
    name = Path(file_storage.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise LoadError(f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXT))}")
    data = file_storage.read()
    if not data:
        raise LoadError("The file is empty.")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise LoadError(f"File exceeds the {config.MAX_FILE_MB} MB limit.")
    file_id = uuid.uuid4().hex
    path = config.UPLOAD_DIR / f"{file_id}{ext}"
    path.write_bytes(data)
    return {"id": file_id, "name": name, "ext": ext, "path": str(path), "size_bytes": len(data)}

def _read_text_table(path):
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise LoadError("Could not detect a table in this text file.")

def _read_pdf(path):
    try:
        import pdfplumber
    except ImportError:
        raise LoadError("PDF support needs 'pdfplumber' installed.")
    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                rows.extend(table)
    if len(rows) < 2:
        raise LoadError("No readable tables found in this PDF.")
    header, body = rows[0], rows[1:]
    return pd.DataFrame(body, columns=[str(h) if h else f"col_{i}" for i, h in enumerate(header)])

def load_dataframe(path):
    ext = Path(path).suffix.lower()
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".json":
            obj = json.loads(Path(path).read_text(encoding="utf-8"))
            df = pd.json_normalize(obj)
        elif ext == ".txt":
            df = _read_text_table(path)
        elif ext == ".pdf":
            df = _read_pdf(path)
        else:
            raise LoadError("Unsupported format.")
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(f"Could not read file: {e}")
    if df.empty or df.shape[1] == 0:
        raise LoadError("The dataset is empty.")
    if len(df) > config.MAX_ROWS:
        raise LoadError(f"Dataset has more than {config.MAX_ROWS:,} rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df
