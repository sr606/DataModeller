import json
import os
import re
from typing import Any, Dict, List, Tuple

from openai import AzureOpenAI
from graphviz import Source
from graphviz.backend import ExecutableNotFound

from agent import (
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    Stage,
    build_global_links,
    compute_complexity,
    detect_undefined_references,
    longest_chain,
    parse_stages,
    sanitize_id,
)


def _load_local_env() -> None:
    # Load .env from current working directory or script directory.
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
    ]
    seen = set()
    for env_path in candidates:
        norm = os.path.abspath(env_path)
        if norm in seen:
            continue
        seen.add(norm)
        if not os.path.isfile(norm):
            continue
        try:
            with open(norm, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if (
                        len(value) >= 2
                        and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
                    ):
                        value = value[1:-1]
                    if key:
                        os.environ.setdefault(key, value)
        except Exception:
            # Non-fatal: agent still validates required variables explicitly.
            continue


def _extract_json_object(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class LineageAgent:
    """
    Hybrid Semantic Lineage Agent:
    - Deterministic tools for structure and graph integrity
    - LLM planner + enrichment for hard semantic tasks
    """

    def __init__(self) -> None:
        self.client = self._init_client()
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if not self.deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is required for LineageAgent.")
        self.tools = {
            "extract_structure": self.extract_structure,
            "sql_analysis": self.sql_analysis,
            "confidence_scoring": self.confidence_scoring,
            "llm_enrichment": self.llm_enrichment,
            "llm_global_semantics": self.llm_global_semantics,
            "validate": self.validate,
            "generate_diagram": self.generate_diagram,
        }

    @staticmethod
    def _init_client() -> AzureOpenAI:
        _load_local_env()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        if not api_key or not api_version or not endpoint:
            raise RuntimeError(
                "Azure OpenAI credentials are required: "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT."
            )
        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> Dict[str, Any]:
        last_text = ""
        for _ in range(3):
            resp = self.client.chat.completions.create(
                model=self.deployment,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            last_text = (resp.choices[0].message.content if resp.choices else "") or ""
            obj = _extract_json_object(last_text)
            if obj:
                return obj

            # Self-correction turn.
            repair = self.client.chat.completions.create(
                model=self.deployment,
                temperature=0,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": "Return valid JSON only. No explanation."},
                    {
                        "role": "user",
                        "content": f"Repair this to valid JSON object only:\n{last_text}",
                    },
                ],
            )
            repaired = (repair.choices[0].message.content if repair.choices else "") or ""
            obj = _extract_json_object(repaired)
            if obj:
                return obj
        raise RuntimeError(f"LLM JSON parsing failed. Last response: {last_text[:300]}")

    def plan(self, input_text: str) -> Dict[str, Any]:
        # Quick deterministic metadata for planner.
        raw_stages = parse_stages(input_text)
        sql_lengths = []
        joins = 0
        max_depth = 0
        for s in raw_stages:
            sql_block = ""
            m = re.search(r"(?is)SQL:\s*(.*)", s.block)
            if m:
                sql_block = m.group(1)
            sql_lengths.append(len(sql_block))
            joins += len(s.join_blocks)
            depth = 0
            dmax = 0
            for ch in sql_block:
                if ch == "(":
                    depth += 1
                    dmax = max(dmax, depth)
                elif ch == ")":
                    depth = max(0, depth - 1)
            max_depth = max(max_depth, dmax)

        metadata = {
            "stage_count": len(raw_stages),
            "sql_lengths": sql_lengths[:50],
            "join_count": joins,
            "max_sql_nesting_depth": max_depth,
        }

        prompt = (
            "Plan tool execution for ETL lineage extraction.\n"
            "Tools: extract_structure, sql_analysis, confidence_scoring, llm_enrichment, llm_global_semantics, validate, generate_diagram\n"
            "Return strict JSON:\n"
            "{\n"
            '  "tool_sequence": ["..."],\n'
            '  "needs_llm_enhancement": true,\n'
            '  "llm_trigger_reason": "short reason",\n'
            '  "confidence_threshold": 0.75\n'
            "}\n"
            "Always include generate_diagram last. Use llm_enrichment when complexity is high or ambiguous."
        )
        obj = self.chat_json("Return JSON only.", f"{prompt}\n\n{json.dumps(metadata)}", max_tokens=350)

        seq = obj.get("tool_sequence", [])
        if not isinstance(seq, list):
            seq = []
        seq = [x for x in seq if x in self.tools]
        if "extract_structure" not in seq:
            seq.insert(0, "extract_structure")
        if "validate" not in seq:
            seq.append("validate")
        if "llm_global_semantics" not in seq:
            seq.insert(max(1, len(seq) - 2), "llm_global_semantics")
        if "generate_diagram" not in seq:
            seq.append("generate_diagram")

        needs = bool(obj.get("needs_llm_enhancement", True))
        thr = obj.get("confidence_threshold", 0.75)
        try:
            thr = float(thr)
        except Exception:
            thr = 0.75
        thr = _clamp(thr, 0.5, 0.95)

        return {
            "tool_sequence": seq,
            "needs_llm_enhancement": needs,
            "llm_trigger_reason": str(obj.get("llm_trigger_reason", "semantic ambiguity")),
            "confidence_threshold": thr,
        }

    def extract_structure(self, state: Dict[str, Any]) -> None:
        text = state["input_text"]
        stages = parse_stages(text)
        for s in stages:
            if not hasattr(s, "llm_annotation"):
                setattr(s, "llm_annotation", "")
            if not hasattr(s, "llm_rules"):
                setattr(s, "llm_rules", [])
            if not hasattr(s, "llm_constraint_labels"):
                setattr(s, "llm_constraint_labels", {})
            detect_undefined_references(s)
        state["stages"] = stages
        state["links"] = build_global_links(stages)

    def sql_analysis(self, state: Dict[str, Any]) -> None:
        stages: List[Stage] = state["stages"]
        sql_complex = []
        total_join_tokens = 0
        for s in stages:
            sql_text = self._extract_stage_sql(s.block)
            sql_len = len(re.findall(r"\bSELECT\b", sql_text, re.IGNORECASE))
            depth = 0
            dmax = 0
            for ch in sql_text:
                if ch == "(":
                    depth += 1
                    dmax = max(dmax, depth)
                elif ch == ")":
                    depth = max(0, depth - 1)
            join_tokens = len(
                re.findall(
                    r"(?is)\b(?:left|right|full|inner|cross)?(?:\s+outer)?\s+join\b|\bjoin\b",
                    sql_text,
                )
            )
            total_join_tokens += join_tokens
            sql_complex.append(
                {"stage": s.name, "select_count": sql_len, "nest_depth": dmax, "join_tokens": join_tokens}
            )
        state["sql_complexity"] = sql_complex
        state["detected_join_count"] = total_join_tokens

    @staticmethod
    def _extract_stage_sql(block: str) -> str:
        m = re.search(
            r"(?is)SQL:\s*(.*?)(?=^\s*(StageType:|Output:|Input:|Transformations:|Constraint\s*\(|//\s*---|\Z))",
            block,
            re.MULTILINE,
        )
        return (m.group(1).strip() if m else "")

    def _stage_confidence(self, stage: Stage) -> float:
        score = 1.0
        unresolved = sum(1 for t in stage.transformations if "Unresolved Mapping" in t.issue)
        undefined = sum(1 for t in stage.transformations if "Undefined Reference" in t.issue)
        score -= min(0.35, unresolved * 0.02)
        score -= min(0.35, undefined * 0.03)
        if stage.stage_type == "Unknown":
            score -= 0.2
        if stage.constraints and not stage.output_links:
            score -= 0.1
        if len(stage.join_blocks) > 8:
            score -= 0.1
        return _clamp(score, 0.0, 1.0)

    @staticmethod
    def _extract_stagevars_multiline(block: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        section = re.search(
            r"(?is)Stage Variables:\s*(.*?)(?=^\s*(Constraint\s*\(|StageType:|Transformations:|Output:|Input:|Link File|//\s*---|\Z))",
            block,
            re.MULTILINE,
        )
        if not section:
            return out
        text = section.group(1)
        pat = re.compile(
            r"(?is)StageVar\s+([A-Za-z_]\w*)\s*=\s*(.*?)(?=^\s*StageVar\s+[A-Za-z_]\w*\s*=|^\s*(?:Constraint\s*\(|StageType:|Transformations:|Output:|Input:|Link File|//\s*---|\Z))",
            re.MULTILINE,
        )
        for m in pat.finditer(text):
            name = m.group(1).strip()
            expr = re.sub(r"\s+", " ", m.group(2)).strip()
            if name and expr:
                out[name] = expr
        return out

    @staticmethod
    def _cluster_stagevars(stagevars: Dict[str, str]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[str]] = {
            "Eligibility": [],
            "Live Status": [],
            "Transition": [],
            "Aggregation": [],
            "Lookup Checks": [],
            "Other": [],
        }
        for name, expr in stagevars.items():
            key = f"{name} {expr}".lower()
            if any(x in key for x in ["canquote", "canorder", "authtype", "termveh", "callbar"]):
                groups["Eligibility"].append(name)
            elif any(x in key for x in ["liveorder", "liveveh", "quotecount", "authquote"]):
                groups["Live Status"].append(name)
            elif any(x in key for x in ["newneeds", "droppedneeds", "flagchange"]):
                groups["Transition"].append(name)
            elif any(x in key for x in ["reportingflag", "svreportingflag", "aggregation"]):
                groups["Aggregation"].append(name)
            elif any(x in key for x in ["notfound", "lkp", "lookup"]):
                groups["Lookup Checks"].append(name)
            else:
                groups["Other"].append(name)
        out: List[Dict[str, Any]] = []
        for k, vals in groups.items():
            if vals:
                out.append({"category": k, "stagevars": vals})
        return out

    @staticmethod
    def _chunk_sql_by_sections(sql_text: str, max_chunk: int = 2600) -> List[str]:
        if not sql_text:
            return []
        # Semantic split by SQL section keywords at line/word boundaries.
        parts = re.split(
            r"(?is)(?=\bSELECT\b|\bFROM\b|\bLEFT\s+OUTER\s+JOIN\b|\bLEFT\s+JOIN\b|\bINNER\s+JOIN\b|\bRIGHT\s+JOIN\b|\bCROSS\s+JOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b)",
            sql_text,
        )
        parts = [p.strip() for p in parts if p and p.strip()]
        chunks: List[str] = []
        cur = ""
        for p in parts:
            if len(cur) + len(p) + 1 <= max_chunk:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks[:8]

    def confidence_scoring(self, state: Dict[str, Any]) -> None:
        stages: List[Stage] = state["stages"]
        conf = {s.name: self._stage_confidence(s) for s in stages}
        state["stage_confidence"] = conf
        state["overall_confidence"] = sum(conf.values()) / len(conf) if conf else 0.0

    def llm_enrichment(self, state: Dict[str, Any]) -> None:
        stages: List[Stage] = state["stages"]
        threshold = float(state.get("plan", {}).get("confidence_threshold", 0.75))
        conf = state.get("stage_confidence", {})
        enrich_events: List[Dict[str, Any]] = []

        for s in stages:
            if not hasattr(s, "llm_annotation"):
                setattr(s, "llm_annotation", "")
            if not hasattr(s, "llm_rules"):
                setattr(s, "llm_rules", [])
            if not hasattr(s, "llm_constraint_labels"):
                setattr(s, "llm_constraint_labels", {})
            sql_text = self._extract_stage_sql(s.block)
            has_sql = bool(sql_text.strip())
            setattr(s, "has_sql", has_sql)
            low_conf = conf.get(s.name, 1.0) < threshold
            stagevars_full = self._extract_stagevars_multiline(s.block)
            setattr(s, "stage_vars_full", stagevars_full)
            sv_clusters = self._cluster_stagevars(stagevars_full)
            setattr(s, "stage_var_clusters", sv_clusters)
            complex_stage = (
                len(s.join_blocks) > 4
                or len(s.constraints) > 0
                or len(stagevars_full) > 5
                or "hashedfile" in s.stage_type.lower()
                or has_sql
            )
            if not (low_conf or complex_stage):
                continue

            payload = {
                "stage_name": s.name,
                "stage_type": s.stage_type,
                "inputs": [x[2] or x[1] for x in s.inputs],
                "outputs": [x[1] for x in s.outputs],
                "constraints": [c.expression for c in s.constraints],
                "transformations": [{"target": t.target_col, "expr": t.expression} for t in s.transformations[:50]],
                "stage_variables": [{"name": k, "logic": v[:500]} for k, v in list(stagevars_full.items())[:30]],
                "stagevar_clusters_seed": sv_clusters,
            }
            prompt = (
                "Summarize stage semantics for lineage with structural reasoning.\n"
                "Return strict JSON:\n"
                "{\n"
                '  "annotation": "short role",\n'
                '  "rules": ["up to 3 business rules"],\n'
                '  "constraint_labels": {"exact_constraint_expression": "short decision label"},\n'
                '  "rule_clusters": [\n'
                '    {"rule_name":"...", "category":"...", "related_stagevars":["..."], "description":"..."}\n'
                "  ],\n"
                '  "lookup_type": "dimension|rule_table|fact_reference|temp_stage|Unknown"\n'
                "}\n"
                "If no stage variables exist, rule_clusters may be empty.\n"
                "Do not invent missing logic. Use Unknown when unclear."
            )
            obj = self.chat_json("Return JSON only.", f"{prompt}\n\n{json.dumps(payload)}", max_tokens=500)
            ann = str(obj.get("annotation", "")).strip()
            if ann and ann.lower() != "unknown":
                s.llm_annotation = ann
            rules = obj.get("rules", [])
            if isinstance(rules, list):
                s.llm_rules = [str(x).strip() for x in rules if str(x).strip() and str(x).strip().lower() != "unknown"][:3]
            cmap = obj.get("constraint_labels", {})
            if isinstance(cmap, dict):
                for k, v in cmap.items():
                    if str(k).strip() and str(v).strip():
                        s.llm_constraint_labels[str(k).strip()] = str(v).strip()
            clusters = obj.get("rule_clusters", [])
            if isinstance(clusters, list):
                setattr(s, "llm_rule_clusters", clusters[:8])
            lkt = str(obj.get("lookup_type", "")).strip()
            if lkt:
                setattr(s, "llm_lookup_type", lkt)

            # Dedicated SQL semantics pass only for SQL-bearing stages.
            if has_sql:
                sql_chunks = self._chunk_sql_by_sections(sql_text)
                sql_payload = {
                    "stage_name": s.name,
                    "sql_chunks": sql_chunks,
                    "join_tokens_detected": len(re.findall(r"(?is)\\b(?:left|right|full|inner|cross)?(?:\\s+outer)?\\s+join\\b|\\bjoin\\b", sql_text)),
                }
                sql_prompt = (
                    "Analyze SQL semantics. Return strict JSON:\n"
                    "{\n"
                    '  "semantic_role": "extraction|enrichment|pivot|change_detection|aggregation|rule_evaluation|Unknown",\n'
                    '  "estimated_join_complexity": "low|medium|high|Unknown",\n'
                    '  "contains_row_expansion": true,\n'
                    '  "contains_historical_comparison": true,\n'
                    '  "contains_change_detection": true,\n'
                    '  "contains_aggregation": true,\n'
                    '  "contains_lookup_enrichment": true,\n'
                    '  "sql_intent_summary": "short summary"\n'
                    "}"
                )
                try:
                    sql_obj = self.chat_json("Return JSON only.", f"{sql_prompt}\n\n{json.dumps(sql_payload)}", max_tokens=420)
                    setattr(s, "llm_sql_semantics", sql_obj if isinstance(sql_obj, dict) else {})
                except Exception:
                    setattr(s, "llm_sql_semantics", {})
            else:
                setattr(s, "llm_sql_semantics", {})

            enrich_events.append(
                {
                    "stage": s.name,
                    "confidence": round(conf.get(s.name, 1.0), 4),
                    "trigger": "low_confidence" if low_conf else "complex_stage",
                    "annotation": s.llm_annotation,
                    "rules": s.llm_rules,
                    "constraint_labels": s.llm_constraint_labels,
                    "sql_semantics": getattr(s, "llm_sql_semantics", {}),
                    "rule_clusters": getattr(s, "llm_rule_clusters", []),
                    "lookup_type": getattr(s, "llm_lookup_type", ""),
                    "stagevar_clusters_seed": sv_clusters,
                }
            )
        state["enrichment_audit"] = enrich_events

    def llm_global_semantics(self, state: Dict[str, Any]) -> None:
        stages: List[Stage] = state["stages"]
        links: List[Tuple[str, str, str]] = state["links"]
        payload = {
            "stages": [{"name": s.name, "type": s.stage_type, "lookup_type": getattr(s, "llm_lookup_type", "")} for s in stages],
            "edges": [{"from": a, "to": b, "link": c} for a, b, c in links],
        }
        prompt = (
            "Infer ETL job-level semantics from stage graph topology and stage semantics.\n"
            "Return strict JSON:\n"
            "{\n"
            '  "job_type": "specific short phrase",\n'
            '  "primary_purpose": "specific sentence",\n'
            '  "key_logic_areas": ["business logic area names"],\n'
            '  "complexity_level": "Low|Medium|High",\n'
            '  "has_historical_dependency": true,\n'
            '  "has_row_expansion": true,\n'
            '  "decision_depth": "Shallow|Moderate|Deep",\n'
            '  "logical_subflows": [{"name":"...", "stages":["..."]}],\n'
            '  "primary_pipeline": ["ordered stage names"],\n'
            '  "secondary_pipeline": ["ordered stage names"]\n'
            "}\n"
            "Do not invent stage names."
        )
        try:
            obj = self.chat_json("Return JSON only.", f"{prompt}\n\n{json.dumps(payload)}", max_tokens=650)
            state["job_semantics"] = {
                "job_type": str(obj.get("job_type", "Unknown")).strip() or "Unknown",
                "primary_purpose": str(obj.get("primary_purpose", "Unknown")).strip() or "Unknown",
                "key_logic_areas": obj.get("key_logic_areas", []) if isinstance(obj.get("key_logic_areas", []), list) else [],
                "complexity_level": str(obj.get("complexity_level", "Unknown")).strip() or "Unknown",
                "has_historical_dependency": bool(obj.get("has_historical_dependency", False)),
                "has_row_expansion": bool(obj.get("has_row_expansion", False)),
                "decision_depth": str(obj.get("decision_depth", "Unknown")).strip() or "Unknown",
                "logical_subflows": obj.get("logical_subflows", []) if isinstance(obj.get("logical_subflows", []), list) else [],
                "primary_pipeline": obj.get("primary_pipeline", []) if isinstance(obj.get("primary_pipeline", []), list) else [],
                "secondary_pipeline": obj.get("secondary_pipeline", []) if isinstance(obj.get("secondary_pipeline", []), list) else [],
            }
        except Exception:
            # Deterministic fallback summary if LLM global semantics fails.
            state["job_semantics"] = {
                "job_type": "ETL Lineage Pipeline",
                "primary_purpose": "Transform and route records across staged datasets",
                "key_logic_areas": [],
                "complexity_level": "Unknown",
                "has_historical_dependency": False,
                "has_row_expansion": False,
                "decision_depth": "Unknown",
                "logical_subflows": [],
                "primary_pipeline": [],
                "secondary_pipeline": [],
            }

    def validate(self, state: Dict[str, Any]) -> None:
        stages: List[Stage] = state["stages"]
        complexity_score, transformed_cols = compute_complexity(stages)
        state["complexity_score"] = complexity_score
        state["transformed_columns"] = transformed_cols
        if complexity_score > 150:
            state["final_output"] = "Job too complex for single detailed diagram. Recommend stage-wise lineage generation."

    def _layer_for_stage(self, stage: Stage, has_upstream: bool, has_downstream: bool, include_lookup: bool) -> str:
        st = stage.stage_type.lower()
        if include_lookup and "hashedfile" in st:
            return "lookup"
        if "transformer" in st:
            return "transformation"
        if "oracleconnector" in st:
            if not has_upstream:
                return "source"
            if has_downstream:
                return "transformation"
            return "target"
        if "seqfile" in st:
            return "target"
        if not has_upstream:
            return "source"
        if not has_downstream:
            return "target"
        return "transformation"

    def _build_arch_dot(
        self,
        graph_name: str,
        stages: List[Stage],
        links: List[Tuple[str, str, str]],
        include_lookup: bool,
        include_decisions: bool,
        job_semantics: Dict[str, Any] | None = None,
        detected_join_count: int | None = None,
    ) -> str:
        # Canonicalize stage names to eliminate duplicate nodes caused by case/spacing variants.
        by_key: Dict[str, Stage] = {}
        for s in stages:
            key = re.sub(r"\s+", " ", s.name.strip()).lower()
            if key not in by_key:
                by_key[key] = s
        uniq_stages: List[Stage] = list(by_key.values())
        name_to_canonical = {
            re.sub(r"\s+", " ", s.name.strip()).lower(): s.name for s in uniq_stages
        }

        def canon(name: str) -> str:
            key = re.sub(r"\s+", " ", (name or "").strip()).lower()
            return name_to_canonical.get(key, name)

        # Split entities: processing stages vs storage stages (hashed files).
        storage_keys = {
            s.name for s in uniq_stages if "hashedfile" in s.stage_type.lower()
        }
        process_stages = [s for s in uniq_stages if s.name not in storage_keys]
        all_entity_names = [s.name for s in uniq_stages]
        node_ids = {n: sanitize_id(n, f"n_{i}") for i, n in enumerate(all_entity_names, 1)}

        downstream = {s.name: 0 for s in uniq_stages}
        upstream = {s.name: 0 for s in uniq_stages}
        canon_links: List[Tuple[str, str, str]] = []
        for src, dst, ds in links:
            csrc = canon(src)
            cdst = canon(dst)
            canon_links.append((csrc, cdst, ds))
            if csrc in downstream:
                downstream[csrc] += 1
            if cdst in upstream:
                upstream[cdst] += 1

        stage_layer = {
            s.name: self._layer_for_stage(
                s,
                has_upstream=upstream.get(s.name, 0) > 0,
                has_downstream=downstream.get(s.name, 0) > 0,
                include_lookup=include_lookup,
            )
            for s in process_stages
        }

        lines = [
            f'digraph {sanitize_id(graph_name, "lineage_graph")} {{',
            "rankdir=LR;",
            'fontsize=10;',
            'fontname="Arial";',
            'node [fontname="Arial", fontsize=10, style=filled];',
            'edge [fontname="Arial", fontsize=8];',
            "",
            "subgraph cluster_source {",
            'label="Source Layer";',
            "style=rounded;",
            "color=grey;",
        ]
        declared_nodes = set()
        for s in process_stages:
            if stage_layer.get(s.name) == "source":
                note = getattr(s, "llm_annotation", "") or ""
                lbl = s.name if not note else f"{s.name}\n({note})"
                nid = node_ids[s.name]
                if nid not in declared_nodes:
                    lines.append(f'{nid} [label="{_dot_escape(lbl)}", shape=cylinder, fillcolor="#FFD1DC"];')
                    declared_nodes.add(nid)
        lines.append("}")

        if include_lookup:
            lines.extend(
                [
                    "",
                    "subgraph cluster_lookup {",
                    'label="Lookup / Hashed Layer";',
                    "style=rounded;",
                    "color=blue;",
                ]
            )
            for s in uniq_stages:
                if s.name in storage_keys or stage_layer.get(s.name) == "lookup":
                    lkt = getattr(s, "llm_lookup_type", "")
                    lbl = s.name if not lkt or lkt.lower() == "unknown" else f"{s.name}\n({lkt})"
                    nid = node_ids[s.name]
                    if nid not in declared_nodes:
                        lines.append(f'{nid} [label="{_dot_escape(lbl)}", shape=cylinder, fillcolor="#E1F5FE"];')
                        declared_nodes.add(nid)
            lines.append("}")

        lines.extend(
            [
                "",
                "subgraph cluster_processing {",
                'label="Transformation Layer";',
                "style=rounded;",
                "color=black;",
            ]
        )
        for s in process_stages:
            if stage_layer.get(s.name) == "transformation":
                ann = getattr(s, "llm_annotation", "") or ""
                rules = getattr(s, "llm_rules", [])[:2]
                label_lines = [s.name]
                sql_sem = getattr(s, "llm_sql_semantics", {})
                if isinstance(sql_sem, dict):
                    role = str(sql_sem.get("semantic_role", "")).strip() if bool(getattr(s, "has_sql", False)) else ""
                    if role and role.lower() != "unknown":
                        label_lines.append(f"- SQL role: {role}")
                    if bool(sql_sem.get("contains_row_expansion")):
                        label_lines.append("- Row expansion detected")
                    if bool(sql_sem.get("contains_change_detection")):
                        label_lines.append("- Difference detection present")
                    intent = str(sql_sem.get("sql_intent_summary", "")).strip()
                    if intent and intent.lower() != "unknown":
                        label_lines.append(f"- {intent}")
                if ann:
                    label_lines.append(f"- {ann}")
                for r in rules:
                    label_lines.append(f"- {r}")
                clusters = getattr(s, "llm_rule_clusters", [])
                if isinstance(clusters, list) and clusters:
                    label_lines.append(f"Rule Clusters: {len(clusters)}")
                if s.constraints:
                    label_lines.append(f"Constraints: {len(s.constraints)}")
                if s.stage_vars_defined:
                    label_lines.append(f"Stage Variables: {len(s.stage_vars_defined)}")
                nid = node_ids[s.name]
                if nid not in declared_nodes:
                    lines.append(
                        f'{nid} [label="{_dot_escape(chr(10).join(label_lines))}", shape=box, fillcolor="#FFF9C4"];'
                    )
                    declared_nodes.add(nid)
        lines.append("}")

        lines.extend(
            [
                "",
                "subgraph cluster_target {",
                'label="Target Layer";',
                "style=rounded;",
                "color=green;",
            ]
        )
        for s in process_stages:
            if stage_layer.get(s.name) == "target":
                st = s.stage_type.lower()
                shape = "note" if ("seqfile" in st or "exception" in s.name.lower()) else "cylinder"
                nid = node_ids[s.name]
                if nid not in declared_nodes:
                    lines.append(f'{nid} [label="{_dot_escape(s.name)}", shape={shape}, fillcolor="#C8E6C9"];')
                    declared_nodes.add(nid)
        lines.append("}")

        # Core lineage edges
        seen = set()
        for src, dst, ds in canon_links:
            # In high-level view, collapse storage nodes to reduce clutter.
            if not include_lookup and (src in storage_keys or dst in storage_keys):
                continue
            src_id = node_ids.get(src, "Unknown")
            dst_id = node_ids.get(dst)
            if not dst_id:
                continue
            if src_id == "Unknown":
                lines.append('Unknown [label="Unknown", shape=cylinder, fillcolor="#F5B7B1"];')
            key = (src_id, dst_id, ds)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f'{src_id} -> {dst_id} [label="{_dot_escape(ds or "Lineage")}"];')

        if include_decisions:
            input_link_to_stage: Dict[str, List[str]] = {}
            for s in process_stages:
                for _, _, link_name in s.inputs:
                    if link_name:
                        input_link_to_stage.setdefault(link_name, []).append(canon(s.name))

            for s in process_stages:
                if not s.constraints:
                    continue
                sid = node_ids[s.name]
                labels = getattr(s, "llm_constraint_labels", {})
                for idx, c in enumerate(s.constraints, 1):
                    # Route constraints to known output-link consumers.
                    targets = [t for t in input_link_to_stage.get(c.name, []) if t != s.name]
                    if not targets:
                        continue
                    label = labels.get(c.expression) or ("Reporting Flag = Y?" if "svreportingflag" in c.expression.lower() else c.expression[:45] + ("..." if len(c.expression) > 45 else ""))
                    did = f"{sid}_decision_{idx}"
                    lines.append(f'{did} [shape=diamond, label="{_dot_escape(label)}", fillcolor="#FADBD8"];')
                    lines.append(f'{sid} -> {did} [label="Constraint"];')
                    for t in targets:
                        lines.append(f'{did} -> {node_ids[t]} [label="Route when true", color=blue];')

        source_count = sum(1 for s in process_stages if stage_layer.get(s.name) == "source")
        decision_count = sum(len(s.constraints) for s in process_stages)
        js = job_semantics or {}
        job_type = str(js.get("job_type", "Unknown"))
        purpose = str(js.get("primary_purpose", "Unknown"))
        logic_areas = js.get("key_logic_areas", []) if isinstance(js.get("key_logic_areas", []), list) else []
        complexity_level = str(js.get("complexity_level", "Unknown"))
        # Derive from stage semantics to prevent global-summary contradictions.
        row_exp_detected = any(
            bool(getattr(s, "llm_sql_semantics", {}).get("contains_row_expansion", False)) for s in process_stages
        )
        hist_dep_detected = any(
            bool(getattr(s, "llm_sql_semantics", {}).get("contains_historical_comparison", False)) for s in process_stages
        )
        hist_dep = "Yes" if (bool(js.get("has_historical_dependency", False)) or hist_dep_detected) else "No"
        row_exp = "Yes" if (bool(js.get("has_row_expansion", False)) or row_exp_detected) else "No"
        decision_depth = str(js.get("decision_depth", "Unknown"))
        subflows = js.get("logical_subflows", []) if isinstance(js.get("logical_subflows", []), list) else []
        primary_pipeline = js.get("primary_pipeline", []) if isinstance(js.get("primary_pipeline", []), list) else []
        secondary_pipeline = js.get("secondary_pipeline", []) if isinstance(js.get("secondary_pipeline", []), list) else []
        summary = (
            "Semantic Summary:\n"
            f"- Job Type: {job_type}\n"
            f"- Purpose: {purpose}\n"
            f"- Complexity Level: {complexity_level}\n"
            f"- Historical Dependency: {hist_dep}\n"
            f"- Row Expansion: {row_exp}\n"
            f"- Decision Depth: {decision_depth}\n"
            f"- Source Stages: {source_count}\n"
            f"- Constraint Routes: {decision_count}"
        )
        if logic_areas:
            summary += "\n- Logic Areas: " + ", ".join(str(x) for x in logic_areas[:4])
        lines.append(f'Note1 [shape=plaintext, label="{_dot_escape(summary)}", fontsize=8, fillcolor="white"];')
        if subflows:
            sub_txt = "Subflows:\n" + "\n".join(
                f"- {str(sf.get('name', 'Flow'))}" for sf in subflows[:3] if isinstance(sf, dict)
            )
            lines.append(f'Note2 [shape=plaintext, label="{_dot_escape(sub_txt)}", fontsize=8, fillcolor="white"];')
        if primary_pipeline or secondary_pipeline:
            pipe_txt = "Pipelines:\n"
            if primary_pipeline:
                pipe_txt += "- Primary: " + " -> ".join(str(x) for x in primary_pipeline[:8]) + "\n"
            if secondary_pipeline:
                pipe_txt += "- Secondary: " + " -> ".join(str(x) for x in secondary_pipeline[:8])
            lines.append(f'Note3 [shape=plaintext, label="{_dot_escape(pipe_txt.strip())}", fontsize=8, fillcolor="white"];')
        lines.append("}")
        return "\n".join(lines)

    def generate_diagram(self, state: Dict[str, Any]) -> None:
        if state.get("final_output"):
            return
        stages: List[Stage] = state["stages"]
        links = state["links"]
        graph_name = sanitize_id(state["file_stem"], "lineage")
        chain_len = longest_chain([(a, b) for a, b, _ in links])
        _rankdir = "TB" if (len(stages) > 10 or chain_len > 8) else "LR"

        # Architecture-first rendering specialized for large DataStage jobs.
        high = self._build_arch_dot(
            graph_name,
            stages,
            links,
            include_lookup=False,
            include_decisions=False,
            job_semantics=state.get("job_semantics", {}),
            detected_join_count=state.get("detected_join_count"),
        )
        detailed = self._build_arch_dot(
            graph_name + "_detailed",
            stages,
            links,
            include_lookup=True,
            include_decisions=True,
            job_semantics=state.get("job_semantics", {}),
            detected_join_count=state.get("detected_join_count"),
        )
        state["high_dot"] = high
        state["detailed_dot"] = detailed
        state["final_output"] = "\n".join(
            [
                "BEGIN_OUTPUT",
                "",
                "HIGH LEVEL DOT",
                high,
                "DETAILED DOT",
                detailed,
                "END_OUTPUT",
            ]
        )

    def run(self, input_text: str, file_name: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "input_text": input_text,
            "file_name": file_name,
            "file_stem": os.path.splitext(file_name)[0],
        }
        plan = self.plan(input_text)
        state["plan"] = plan

        for tool_name in plan["tool_sequence"]:
            if tool_name == "llm_enrichment" and not plan.get("needs_llm_enhancement", True):
                continue
            self.tools[tool_name](state)
            if state.get("final_output") and tool_name != "generate_diagram":
                break

        if "final_output" not in state:
            self.generate_diagram(state)
        return state


def run_llm_agent() -> None:
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".txt")])
    if not files:
        print(f"No input files found in {INPUT_FOLDER}")
        return

    agent = LineageAgent()

    def _render_pdf(dot_code: str, output_stem: str) -> None:
        Source(dot_code).render(output_stem, format="pdf", cleanup=True)

    for file_name in files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        base = sanitize_id(os.path.splitext(file_name)[0], "lineage")
        output_path = os.path.join(OUTPUT_FOLDER, f"{base}.dot.txt")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            state = agent.run(text, file_name)
            out = state.get("final_output", "BEGIN_OUTPUT\nParsing error: Unable to extract deterministic lineage from input.\nEND_OUTPUT")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(out)
            audit_path = os.path.join(OUTPUT_FOLDER, f"{base}_audit.json")
            audit_payload = {
                "file": file_name,
                "plan": state.get("plan", {}),
                "overall_confidence": state.get("overall_confidence"),
                "complexity_score": state.get("complexity_score"),
                "transformed_columns": state.get("transformed_columns"),
                "stage_confidence": state.get("stage_confidence", {}),
                "sql_complexity": state.get("sql_complexity", []),
                "llm_enrichment": state.get("enrichment_audit", []),
                "tool_sequence_executed": state.get("plan", {}).get("tool_sequence", []),
            }
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit_payload, f, indent=2)
            if state.get("high_dot") and state.get("detailed_dot"):
                with open(os.path.join(OUTPUT_FOLDER, f"{base}_high_level.dot"), "w", encoding="utf-8") as f:
                    f.write(state["high_dot"])
                with open(os.path.join(OUTPUT_FOLDER, f"{base}_detailed.dot"), "w", encoding="utf-8") as f:
                    f.write(state["detailed_dot"])
                try:
                    _render_pdf(state["high_dot"], os.path.join(OUTPUT_FOLDER, f"{base}_high_level"))
                    _render_pdf(state["detailed_dot"], os.path.join(OUTPUT_FOLDER, f"{base}_detailed"))
                except ExecutableNotFound:
                    print("[WARN] Graphviz 'dot' executable not found. Install Graphviz and add it to PATH to generate PDF.")
                except Exception as ex:
                    print(f"[WARN] PDF render failed: {ex}")
            print(f"Processed {file_name} -> {output_path}")
        except Exception as ex:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("BEGIN_OUTPUT\nParsing error: Unable to extract deterministic lineage from input.\nEND_OUTPUT")
            print(f"Processed {file_name} -> {output_path} [LLM agent fallback due to error: {ex}]")


if __name__ == "__main__":
    run_llm_agent()
