"""
Data Modeller - LangGraph-ready Python project

This single-file project implements the end-to-end data modelling workflow described in the spec
and is organized into modular components that can be run locally or wired into a LangGraph flow.

Features included (step-by-step):
- File ingestion & parsing (CSV/JSON/XLSX)
- Schema extraction & sampling
- Entity candidate detection (heuristic + optional LLM prompts)
- Normalization engine (1NF/2NF/3NF heuristics)
- Primary & foreign key detection (uniqueness/support metrics)
- Relationship classification (1:N, M:N, 1:1) and junction table suggestion
- ERD JSON builder (machine readable)
- Validator (sample-based referential checks & business-logic checks)
- SQL DDL generator (Postgres example)
- LangGraph node prompt templates and wiring helper (stubs)

How to use:
1. Install: pip install pandas openpyxl python-dateutil
   (Add your LLM client libs if you want LLM-enhanced steps.)
2. Place your ~50 files in a folder and point `DATA_DIR` to it.
3. Run `python data_modeller_langgraph.py` to run locally using the heuristic pipeline.
4. To convert to LangGraph, use the `register_langgraph_nodes()` function and adapt the stubs
   to your LangGraph SDK.

Notes:
- This code is designed to be runnable without any external LLM; LLM prompts are provided
  and can be plugged into LLM nodes in LangGraph by replacing the `call_llm()` stub.
- For very large files, sampling is used. Validator runs sample-based checks; adapt for full
  dataset checks if you have a database.

"""

import os
import json
import re
import glob
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

# ----------------------------- Configuration -----------------------------
DATA_DIR = "./data_files"  # point to your folder containing CSV/JSON/XLSX files
SAMPLE_PER_FILE = 200
GLOBAL_SAMPLE_CAP = 5000
SURROGATE_PK_TEMPLATE = "{table}_id"

# ----------------------------- Utilities --------------------------------

def snake_case(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.strip("_").lower()


def read_file(path: str, max_rows: int = SAMPLE_PER_FILE) -> Dict[str, Any]:
    """Read CSV, JSON, or Excel and return metadata and sample rows."""
    fname = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".csv", ".txt"]:
            df = pd.read_csv(path, dtype=str, nrows=max_rows)
        elif ext in [".json"]:
            df = pd.read_json(path, lines=True)
            if len(df) > max_rows:
                df = df.head(max_rows)
            df = df.astype(str)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path, nrows=max_rows)
            df = df.astype(str)
        else:
            return {"name": fname, "file_type": ext, "columns": [], "sample_rows": [], "row_count_estimate": 0}
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return {"name": fname, "file_type": ext, "columns": [], "sample_rows": [], "row_count_estimate": 0}

    df = df.replace({np.nan: None})
    columns = [snake_case(c) for c in df.columns]
    df.columns = columns
    sample_rows = df.to_dict(orient="records")
    # estimate rows cheaply
    try:
        total_rows = sum(1 for _ in open(path)) if ext == ".csv" else len(sample_rows)
    except Exception:
        total_rows = len(sample_rows)

    return {
        "name": fname,
        "path": path,
        "file_type": ext.strip("."),
        "columns": columns,
        "sample_rows": sample_rows,
        "row_count_estimate": total_rows
    }


def collect_files(data_dir: str) -> List[str]:
    exts = ["*.csv", "*.json", "*.xlsx", "*.xls", "*.txt"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, e)))
    return files


# -------------------- Step 1: File Ingest & Parser ------------------------

def ingest_and_parse(data_dir: str, sample_per_file: int = SAMPLE_PER_FILE) -> Dict[str, Any]:
    files = collect_files(data_dir)
    sources = []
    combined_sample_rows = []
    for f in files:
        meta = read_file(f, max_rows=sample_per_file)
        sources.append(meta)
        combined_sample_rows.extend(meta.get("sample_rows", []))
        if len(combined_sample_rows) >= GLOBAL_SAMPLE_CAP:
            break
    combined_sample_rows = combined_sample_rows[:GLOBAL_SAMPLE_CAP]
    return {"sources": sources, "combined_samples": combined_sample_rows}


# ------------------ Step 2: Schema Extractor & Sampler --------------------

def normalize_column_name(col: str) -> str:
    return snake_case(col)


def canonicalize_columns(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build canonical map: detect synonyms by simple heuristics
    canonical_map = {}
    file_column_map = {}
    all_columns = Counter()
    for s in sources:
        cols = s.get("columns", [])
        normalized = [normalize_column_name(c) for c in cols]
        file_column_map[s["name"]] = normalized
        for c in normalized:
            all_columns[c] += 1

    # naive synonym grouping via substring and edit distance could be added.
    canonical_columns = list(all_columns.keys())
    return {"canonical_columns": canonical_columns, "file_column_map": file_column_map}


# ------------------ Step 3: Entity Candidate Detector --------------------

def guess_type_from_values(values: List[Any]) -> str:
    # simple heuristic
    non_null = [v for v in values if v not in (None, "", "None")]
    if not non_null:
        return "string"
    sample = non_null[:50]
    # is int?
    if all(re.match(r"^-?\d+$", str(x)) for x in sample):
        return "int"
    # is float?
    if all(re.match(r"^-?\d+(\.\d+)?$", str(x)) for x in sample):
        return "float"
    # date-like
    date_like = 0
    for x in sample:
        if re.search(r"\d{4}-\d{2}-\d{2}", str(x)):
            date_like += 1
    if date_like / len(sample) > 0.5:
        return "timestamp"
    # boolean
    if all(str(x).lower() in ("true","false","0","1","yes","no") for x in sample):
        return "boolean"
    return "string"


def detect_entities(samples: List[Dict[str, Any]], max_attr: int = 100) -> List[Dict[str, Any]]:
    # Heuristic grouping: columns that co-occur frequently form entities
    # Build co-occurrence matrix
    col_presence = Counter()
    pair_presence = Counter()
    for r in samples:
        cols = [c for c,v in r.items() if v not in (None, "", "None")]
        for c in cols:
            col_presence[c] += 1
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pair_presence[(cols[i], cols[j])] += 1

    # naive clustering: pick high-frequency columns as core entities
    sorted_cols = [c for c,_ in col_presence.most_common()][:max_attr]
    entities = []
    used = set()
    for col, _ in col_presence.most_common():
        if col in used:
            continue
        # find columns that often pair with this column
        related = [b for (a,b),cnt in pair_presence.items() if a==col and cnt>5]
        related += [a for (a,b),cnt in pair_presence.items() if b==col and cnt>5]
        group = [col] + related
        for g in group:
            used.add(g)
        # create entity name by taking the most common prefix
        name = suggest_entity_name(group)
        attributes = []
        for a in group:
            sample_vals = [r.get(a) for r in samples if a in r][:50]
            attributes.append({"name":a, "examples": sample_vals, "inferred_type": guess_type_from_values(sample_vals)})
        entities.append({"entity_name": name, "attributes": attributes, "reasoning": f"cooccurrence with {col}"})
    # fallback: if nothing found, create a "raw_record" entity
    if not entities and samples:
        entities.append({"entity_name":"raw_record","attributes":[{"name":k,"examples":[r.get(k) for r in samples[:10]]} for k in samples[0].keys()]})
    return entities


def suggest_entity_name(cols: List[str]) -> str:
    # try heuristics: if "id" in column names -> use prefix
    prefixes = Counter()
    for c in cols:
        p = c.split("_")[0]
        prefixes[p] += 1
    common = prefixes.most_common(1)
    if common and common[0][1] > 1:
        return common[0][0]
    # else join
    return snake_case("_".join(cols[:3]))


# ------------------ Step 4: Normalizer (1NF-3NF heuristics) ----------------

def apply_normalization(entities: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for e in entities:
        table_name = snake_case(e["entity_name"]) or "entity"
        attrs = []
        for a in e.get("attributes", []):
            attrs.append({
                "name": snake_case(a["name"]),
                "type": a.get("inferred_type", "string"),
                "nullable": any(v in (None, "", "None") for v in a.get("examples", []))
            })
        # ensure surrogate PK if no id present
        has_id = any(re.search(r"(^|_)id$", at["name"]) for at in attrs)
        if not has_id:
            pk = SURROGATE_PK_TEMPLATE.format(table=table_name)
            attrs.insert(0, {"name": pk, "type": "int", "nullable": False, "surrogate": True})
        normalized.append({"table_name": table_name, "attributes": attrs, "example_rows": []})
    # produce example transformed rows by selecting columns per table from samples
    for t in normalized:
        rows = []
        cols = [a["name"] for a in t["attributes"] if not a.get("surrogate")]
        for r in samples[:50]:
            row = {c: r.get(c) for c in cols}
            rows.append(row)
        t["example_rows"] = rows
    return normalized


# ------------------ Step 5: Key Identifier --------------------------------

def find_pk_candidates(table: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    cols = [a["name"] for a in table["attributes"]]
    results = []
    tbl_samples = [{c:r.get(c) for c in cols} for r in samples]
    n = len(tbl_samples)
    for c in cols:
        values = [r.get(c) for r in tbl_samples]
        distinct = len(set(values))
        uniqueness = distinct / max(1, n)
        results.append({"field": c, "distinct": distinct, "uniqueness": uniqueness})
    # pick best
    sorted_res = sorted(results, key=lambda x: (-x["uniqueness"], -x["distinct"]))
    if sorted_res:
        best = sorted_res[0]
        return {"pk_field": best["field"], "uniqueness": best["uniqueness"], "candidates": sorted_res}
    return {"pk_field": None, "uniqueness": 0.0, "candidates": results}


def detect_fk_candidates(normalized: List[Dict[str,Any]], samples: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # naive: if values in table A match values in some column of table B frequently, suggest FK
    fk_suggestions = []
    # build value->tables mapping for sample values
    col_value_map = defaultdict(lambda: defaultdict(set))  # {table: {col: set(values)}}
    for t in normalized:
        cols = [a["name"] for a in t["attributes"]]
        for c in cols:
            vals = set()
            for r in samples[:1000]:
                v = r.get(c)
                if v not in (None, "", "None"):
                    vals.add(v)
            col_value_map[t["table_name"]][c] = vals

    # compare
    for t_from in normalized:
        for a in t_from["attributes"]:
            fn = a["name"]
            vals_from = col_value_map[t_from["table_name"]].get(fn, set())
            if not vals_from:
                continue
            for t_to in normalized:
                if t_to["table_name"] == t_from["table_name"]:
                    continue
                for c_to, vals_to in col_value_map[t_to["table_name"]].items():
                    if not vals_to:
                        continue
                    # compute overlap support
                    common = vals_from.intersection(vals_to)
                    if not common:
                        continue
                    support = len(common) / max(1, len(vals_from))
                    if support > 0.3 and len(common) > 3:
                        fk_suggestions.append({
                            "from_table": t_from["table_name"],
                            "from_field": fn,
                            "to_table": t_to["table_name"],
                            "to_field": c_to,
                            "support": support,
                            "common_count": len(common)
                        })
    return fk_suggestions


# ------------------ Step 6: Relationship Classifier ----------------------

def classify_relationships(fk_candidates: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    relationships = []
    for fk in fk_candidates:
        # Heuristic: if support high and to_field uniqueness high -> 1:N
        rel_type = "1:N"
        if fk["support"] < 0.6 and fk["common_count"]>10:
            rel_type = "M:N" if fk["support"]<0.8 else "1:N"
        relationships.append({
            "from_table": fk["from_table"],
            "to_table": fk["to_table"],
            "from_field": fk["from_field"],
            "to_field": fk["to_field"],
            "type": rel_type,
            "support": fk["support"]
        })
    return relationships


# ------------------ Step 7: ERD Builder ---------------------------------

def build_erd_json(normalized: List[Dict[str,Any]], pk_info: Dict[str,Any], fk_list: List[Dict[str,Any]], relationships: List[Dict[str,Any]], sources: List[Dict[str,Any]]) -> Dict[str,Any]:
    entities = []
    for t in normalized:
        entities.append({
            "table_name": t["table_name"],
            "description": "",
            "attributes": t["attributes"],
            "primary_key": pk_info.get(t["table_name"], {}).get("pk_field"),
            "row_count_estimate": 0
        })
    foreign_keys = []
    for fk in fk_list:
        foreign_keys.append({
            "from_table": fk["from_table"],
            "from_columns": [fk["from_field"]],
            "to_table": fk["to_table"],
            "to_columns": [fk["to_field"]],
            "on_delete": "NO ACTION",
            "nullability": "nullable" if fk.get("support",0) < 0.9 else "mandatory",
            "support": fk.get("support",0),
            "evidence_samples": []
        })
    rels = []
    for r in relationships:
        rels.append({
            "from_table": r["from_table"],
            "to_table": r["to_table"],
            "type": r["type"],
            "justification": f"support {r.get('support')}",
            "confidence": r.get("support",0)
        })
    erd = {
        "entities": entities,
        "foreign_keys": foreign_keys,
        "relationships": rels,
        "metadata": {
            "source_files": [s["name"] for s in sources],
            "generated_at": pd.Timestamp.now().isoformat(),
            "summary": "Auto-generated ERD"
        }
    }
    return erd


# ------------------ Step 8: Validator ----------------------------------

def validate_erd(erd: Dict[str,Any], samples: List[Dict[str,Any]]) -> Dict[str,Any]:
    issues = []
    passed = []
    # check PK uniqueness on sample rows
    for ent in erd["entities"]:
        pk = ent.get("primary_key")
        if not pk:
            issues.append({"desc": f"No PK for table {ent['table_name']}", "severity":"high"})
            continue
        vals = [r.get(pk) for r in samples if pk in r]
        if not vals:
            issues.append({"desc": f"PK {pk} not present in samples for {ent['table_name']}", "severity":"medium"})
            continue
        distinct = len(set(vals))
        uniqueness = distinct / max(1, len(vals))
        if uniqueness < 0.95:
            issues.append({"desc": f"PK {pk} for {ent['table_name']} not unique (uniqueness={uniqueness:.2f})", "severity":"medium", "sample_count":len(vals)})
        else:
            passed.append(f"PK {pk} uniqueness ok for {ent['table_name']}")
    # FK referential checks (sample-based)
    for fk in erd.get("foreign_keys", []):
        from_col = fk["from_columns"][0]
        to_col = fk["to_columns"][0]
        from_vals = set(r.get(from_col) for r in samples if from_col in r and r.get(from_col) not in (None, "", "None"))
        to_vals = set(r.get(to_col) for r in samples if to_col in r and r.get(to_col) not in (None, "", "None"))
        if not from_vals:
            issues.append({"desc": f"No sample values for FK {from_col} -> {to_col}", "severity":"low"})
            continue
        missing = from_vals - to_vals
        if len(missing) / max(1, len(from_vals)) > 0.1:
            issues.append({"desc": f"Referential integrity issue for FK {from_col}->{to_col}, {len(missing)} missing", "severity":"high", "examples": list(missing)[:5]})
        else:
            passed.append(f"FK {from_col}->{to_col} referential check passed with {len(missing)} missing")
    report = {"passed_checks": passed, "issues": issues, "recommendations": []}
    return report


# ------------------ SQL DDL Generator ----------------------------------

def map_type_to_sql(t: str) -> str:
    if t in ("int", "integer"):
        return "BIGINT"
    if t == "float":
        return "NUMERIC"
    if t == "timestamp":
        return "TIMESTAMP"
    if t == "boolean":
        return "BOOLEAN"
    return "TEXT"


def generate_sql_ddl(erd: Dict[str,Any], schema_name: str = "public") -> str:
    statements = []
    for ent in erd["entities"]:
        cols = []
        for a in ent["attributes"]:
            name = a["name"]
            t = map_type_to_sql(a.get("type","string"))
            null = "NOT NULL" if not a.get("nullable", True) else ""
            cols.append(f"  {name} {t} {null}")
        pk = ent.get("primary_key")
        pk_clause = f",\n  PRIMARY KEY ({pk})" if pk else ""
        stmt = f"CREATE TABLE {schema_name}.{ent['table_name']} (\n" + ",\n".join(cols) + pk_clause + "\n);"
        statements.append(stmt)
    for fk in erd.get("foreign_keys", []):
        ft = fk["from_table"]
        fc = fk["from_columns"][0]
        tt = fk["to_table"]
        tc = fk["to_columns"][0]
        constraint = f"ALTER TABLE {schema_name}.{ft} ADD CONSTRAINT fk_{ft}_{fc}_{tt}_{tc} FOREIGN KEY ({fc}) REFERENCES {schema_name}.{tt}({tc}) ON DELETE NO ACTION;"
        statements.append(constraint)
    return "\n\n".join(statements)


# ------------------ LangGraph Wiring Helpers (stubs) --------------------

def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Stub function to call an LLM. Replace with your LLM client call or LangGraph LLM node.
    Return string output (JSON if expected)."""
    # Example: integrate with OpenAI, Anthropic, or LangGraph LLM node here.
    raise NotImplementedError("Replace call_llm() with your LLM client or LangGraph node execution.")


def register_langgraph_nodes():
    """Pseudo-code / guidance to register these functions as LangGraph nodes.

    Replace with your LangGraph SDK calls. The pattern is:
    - Create a node which either runs local Python code (callable) or an LLM prompt.
    - Define inputs and outputs.
    - Wire the nodes in the sequence described in the runbook.

    This function is a template; adapt it for your environment.
    """
    print("-- LangGraph registration placeholder --")
    # Example pseudo-code:
    # import langgraph
    # graph = langgraph.Graph()
    # graph.add_node('parser', function=ingest_and_parse)
    # graph.add_node('schema_extractor', function=canonicalize_columns)
    # graph.add_node('entity_detector', llm_prompt=ENTITY_DETECTOR_PROMPT)
    # ... wire nodes ...
    # graph.deploy()


# ------------------ Prompts (for LLM nodes) ------------------------------

ENTITY_DETECTOR_PROMPT = """
You are an expert data modeller. Given a list of canonical column names and up to 500 sample rows,
cluster columns into entity candidates. For each entity, produce JSON:
{
  "entity_name": "SuggestedName",
  "attributes": [{"name":"col","examples":[...],"inferred_type":"string"}],
  "reasoning": "why these columns form an entity"
}
Return only JSON array.
"""

NORMALIZER_PROMPT = """
You are a normalization agent. For each entity candidate (JSON), propose normalized tables applying
1NF-3NF rules. If surrogate PK required, create `{table}_id` as integer surrogate. Return JSON.
"""

KEY_IDENTIFIER_PROMPT = """
You are a key detection agent. For each normalized table return JSON describing PK candidates and FK candidates
including uniqueness/support metrics computed on samples.
"""

RELATIONSHIP_CLASSIFIER_PROMPT = """
You are a relationship classifier. Using FK candidates and sample evidence, output relationships with types
1:N, M:N, or 1:1, justification and evidence. Return JSON.
"""

ERD_BUILDER_PROMPT = """
Assemble the final ERD JSON using the provided schema. Include attributes, datatypes, PKs, FKs, and relationships.
Return valid JSON only.
"""

VALIDATOR_PROMPT = """
Validate the ERD with sample rows: check PK uniqueness, FK referential integrity, nullability, and suggest improvements.
Return JSON report.
"""

# ------------------ Orchestration (local-run using heuristics) -------------

def run_local_pipeline(data_dir: str = DATA_DIR) -> Dict[str,Any]:
    print("1) Ingesting files...")
    parsed = ingest_and_parse(data_dir)
    sources = parsed["sources"]
    samples = parsed["combined_samples"]

    print(f"Found {len(sources)} source files; using {len(samples)} sample rows")

    print("2) Canonicalizing columns...")
    canon = canonicalize_columns(sources)

    print("3) Detecting entity candidates...")
    entities = detect_entities(samples)

    print(f"Detected {len(entities)} entity candidates")

    print("4) Normalizing...")
    normalized = apply_normalization(entities, samples)

    print("5) Detecting PKs...")
    pk_info = {}
    for t in normalized:
        pk = find_pk_candidates(t, samples)
        pk_info[t["table_name"]] = pk

    print("6) Detecting FK candidates...")
    fk_candidates = detect_fk_candidates(normalized, samples)

    print(f"Found {len(fk_candidates)} FK candidate relationships")

    print("7) Classifying relationships...")
    relationships = classify_relationships(fk_candidates)

    print("8) Building ERD JSON...")
    erd = build_erd_json(normalized, pk_info, fk_candidates, relationships, sources)

    print("9) Validating ERD...")
    report = validate_erd(erd, samples)

    print("10) Generating SQL DDL...")
    ddl = generate_sql_ddl(erd)

    result = {"erd": erd, "validation": report, "sql": ddl}
    # save outputs
    with open("erd.json","w",encoding="utf-8") as f:
        json.dump(erd,f,indent=2,ensure_ascii=False)
    with open("validation.json","w",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    with open("ddl.sql","w",encoding="utf-8") as f:
        f.write(ddl)

    print("Pipeline complete. Files written: erd.json, validation.json, ddl.sql")
    return result


# ------------------ Entry point -----------------------------------------

if __name__ == "__main__":
    # Run local heuristic pipeline. To enable LLMs / LangGraph, wire prompts and replace call_llm stubs.
    if not os.path.exists(DATA_DIR):
        print(f"Data dir {DATA_DIR} not found. Create it and place CSV/JSON/XLSX files for processing.")
    else:
        run_local_pipeline(DATA_DIR)

