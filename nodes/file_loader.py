import os
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path

SUPPORTED = [".csv", ".json", ".xlsx", ".xls"]

def load_file(path: str, sample_rows: int = 50) -> Dict[str, Any]:
    """
    Dynamically load a file (CSV/Excel/JSON) with real-world safety.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = p.suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(
                p,
                dtype=str,
                nrows=sample_rows,
                on_bad_lines="skip",     # 🛠 Fix irregular CSV rows
                engine="python"          # 🛠 Fix comma/quote issues
            )
            df = df.fillna("")
            return {
                "source_type": "csv",
                "columns": list(df.columns),
                "sample_rows": df.head(sample_rows).to_dict(orient="records")
            }

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(p, nrows=sample_rows, dtype=str)
            df = df.fillna("")
            return {
                "source_type": "excel",
                "columns": list(df.columns),
                "sample_rows": df.head(sample_rows).to_dict(orient="records")
            }

        elif ext == ".json":
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)

            # If it's an ERD schema JSON
            if isinstance(obj, dict) and "entities" in obj:
                cols = []
                for e in obj.get("entities", []):
                    for a in e.get("attributes", []):
                        n = a.get("name")
                        if n and n not in cols:
                            cols.append(n)
                return {
                    "source_type": "schema_json",
                    "columns": cols,
                    "sample_rows": [],
                    "raw_text_snippet": json.dumps(obj)[:2000]
                }

            # JSON array → treat as data
            if isinstance(obj, list):
                df = pd.DataFrame(obj).fillna("")
                return {
                    "source_type": "json_data",
                    "columns": list(df.columns),
                    "sample_rows": df.head(sample_rows).to_dict(orient="records")
                }

            # Other JSON object
            return {
                "source_type": "json_other",
                "columns": list(obj.keys()),
                "sample_rows": [],
                "raw_text_snippet": json.dumps(obj)[:2000]
            }

        # fallback
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read(2000)
            return {
                "source_type": "text",
                "columns": [],
                "sample_rows": [],
                "raw_text_snippet": txt
            }

    except Exception as e:
        raise RuntimeError(f"Error reading file {path}: {e}")
