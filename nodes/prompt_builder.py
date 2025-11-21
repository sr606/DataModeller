import json
from typing import Dict, Any

PROMPT_TEMPLATE = """
You are an expert Data Architect and Data Modelling Agent. Perform full end-to-end semantic data modelling (LLM-only).
Input:
- Local file path (present on the executor): {file_url}
- source_type: {source_type}
- Columns: {columns}
- Up to 10 sample rows (JSON): {sample_preview}

Task (output MUST be a single valid JSON object with these top-level keys):
- entities: [{{
    "table_name": "...", "description":"...", 
    "attributes":[{{"name":"", "inferred_type":"", "nullable":true/false, "example_values":[...]}}],
    "primary_key": ["col"] or "col",
    "unique_constraints": [],
    "indexes": [],
    "example_rows":[... up to 5 rows]
}}]
- foreign_keys: [{{"from_table":"", "from_columns":[""], "to_table":"", "to_columns":[""], "nullability":"nullable|mandatory", "support":0.0, "evidence": [...]}}]
- relationships: [{{"from_table":"", "to_table":"", "type":"1:N|M:N|1:1", "justification":"", "confidence":0.0}}]
- sql_ddl: "..." (Postgres DDL statements as a single string)
- validation: {{ "assumptions": [...], "integrity_checks": [...], "confidence": 0.0 }}

Constraints:
- Return JSON only. No extra commentary.
- Use business-friendly names: Customer, Order, OrderItem, Product, Payment, Shipping.
- Create junction tables explicitly for M:N relationships.
- Use SQL types: INTEGER/BIGINT/TEXT/TIMESTAMP/NUMERIC/BOOLEAN.
- Provide brief justifications for relationship choices inside `relationships.justification`.
- If you cannot infer exact values, state assumptions inside `validation.assumptions`.

Begin.
"""

def build_prompt(file_path: str, content_summary: Dict[str, Any]) -> str:
    file_url = f"file://{file_path}"
    columns = content_summary.get("columns", [])
    sample_rows = content_summary.get("sample_rows", [])[:10]
    sample_preview = json.dumps(sample_rows, ensure_ascii=False)
    source_type = content_summary.get("source_type", "unknown")
    return PROMPT_TEMPLATE.format(file_url=file_url, source_type=source_type, columns=columns, sample_preview=sample_preview)
