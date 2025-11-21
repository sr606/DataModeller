import os
import json
from pathlib import Path
from typing import Dict, Any

def write_outputs(parsed: Dict[str, Any], out_dir: str = "output"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    erd_path = os.path.join(out_dir, "erd_openai.json")
    ddl_path = os.path.join(out_dir, "ddl_openai.sql")
    val_path = os.path.join(out_dir, "validation_openai.json")

    with open(erd_path, "w", encoding="utf-8") as f:
        json.dump(parsed.get("entities", parsed), f, indent=2, ensure_ascii=False)

    sql = parsed.get("sql_ddl") or parsed.get("ddl") or parsed.get("sql") or ""
    with open(ddl_path, "w", encoding="utf-8") as f:
        f.write(sql)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(parsed.get("validation", {}), f, indent=2, ensure_ascii=False)
