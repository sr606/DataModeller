import json
import os
import re
from dataclasses import asdict
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
            "Tools: extract_structure, sql_analysis, confidence_scoring, llm_enrichment, validate, generate_diagram\n"
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
        for s in stages:
            sql_len = len(re.findall(r"\bSELECT\b", s.block, re.IGNORECASE))
            depth = 0
            dmax = 0
            for ch in s.block:
                if ch == "(":
                    depth += 1
                    dmax = max(dmax, depth)
                elif ch == ")":
                    depth = max(0, depth - 1)
            sql_complex.append({"stage": s.name, "select_count": sql_len, "nest_depth": dmax})
        state["sql_complexity"] = sql_complex

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
            low_conf = conf.get(s.name, 1.0) < threshold
            complex_stage = len(s.join_blocks) > 4 or len(s.constraints) > 0 or len(s.stage_vars_defined) > 5
            if not (low_conf or complex_stage):
                continue

            payload = {
                "stage_name": s.name,
                "stage_type": s.stage_type,
                "inputs": [x[2] or x[1] for x in s.inputs],
                "outputs": [x[1] for x in s.outputs],
                "constraints": [c.expression for c in s.constraints],
                "transformations": [{"target": t.target_col, "expr": t.expression} for t in s.transformations[:50]],
            }
            prompt = (
                "Summarize stage semantics for lineage.\n"
                "Return strict JSON:\n"
                "{\n"
                '  "annotation": "short role",\n'
                '  "rules": ["up to 3 business rules"],\n'
                '  "constraint_labels": {"exact_constraint_expression": "short decision label"}\n'
                "}\n"
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
            enrich_events.append(
                {
                    "stage": s.name,
                    "confidence": round(conf.get(s.name, 1.0), 4),
                    "trigger": "low_confidence" if low_conf else "complex_stage",
                    "annotation": s.llm_annotation,
                    "rules": s.llm_rules,
                    "constraint_labels": s.llm_constraint_labels,
                }
            )
        state["enrichment_audit"] = enrich_events

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
    ) -> str:
        node_ids = {s.name: sanitize_id(s.name, f"stage_{i}") for i, s in enumerate(stages, 1)}
        downstream = {s.name: 0 for s in stages}
        upstream = {s.name: 0 for s in stages}
        for src, dst, _ in links:
            if src in downstream:
                downstream[src] += 1
            if dst in upstream:
                upstream[dst] += 1

        stage_layer = {
            s.name: self._layer_for_stage(
                s,
                has_upstream=upstream.get(s.name, 0) > 0,
                has_downstream=downstream.get(s.name, 0) > 0,
                include_lookup=include_lookup,
            )
            for s in stages
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
        for s in stages:
            if stage_layer[s.name] == "source":
                note = getattr(s, "llm_annotation", "") or ""
                lbl = s.name if not note else f"{s.name}\n({note})"
                lines.append(f'{node_ids[s.name]} [label="{_dot_escape(lbl)}", shape=cylinder, fillcolor="#FFD1DC"];')
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
            for s in stages:
                if stage_layer[s.name] == "lookup":
                    lines.append(f'{node_ids[s.name]} [label="{_dot_escape(s.name)}", shape=cylinder, fillcolor="#E1F5FE"];')
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
        for s in stages:
            if stage_layer[s.name] == "transformation":
                ann = getattr(s, "llm_annotation", "") or ""
                rules = getattr(s, "llm_rules", [])[:2]
                label_lines = [s.name]
                if ann:
                    label_lines.append(f"- {ann}")
                for r in rules:
                    label_lines.append(f"- {r}")
                if s.constraints:
                    label_lines.append(f"Constraints: {len(s.constraints)}")
                if s.stage_vars_defined:
                    label_lines.append(f"Stage Variables: {len(s.stage_vars_defined)}")
                lines.append(
                    f'{node_ids[s.name]} [label="{_dot_escape(chr(10).join(label_lines))}", shape=box, fillcolor="#FFF9C4"];'
                )
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
        for s in stages:
            if stage_layer[s.name] == "target":
                st = s.stage_type.lower()
                shape = "note" if ("seqfile" in st or "exception" in s.name.lower()) else "cylinder"
                lines.append(f'{node_ids[s.name]} [label="{_dot_escape(s.name)}", shape={shape}, fillcolor="#C8E6C9"];')
        lines.append("}")

        # Core lineage edges
        seen = set()
        for src, dst, ds in links:
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
            for s in stages:
                for _, _, link_name in s.inputs:
                    if link_name:
                        input_link_to_stage.setdefault(link_name, []).append(s.name)

            for s in stages:
                if not s.constraints:
                    continue
                sid = node_ids[s.name]
                labels = getattr(s, "llm_constraint_labels", {})
                for idx, c in enumerate(s.constraints, 1):
                    # Route constraints to known output-link consumers.
                    targets = input_link_to_stage.get(c.name, [])
                    if not targets:
                        continue
                    label = labels.get(c.expression) or ("Reporting Flag = Y?" if "svreportingflag" in c.expression.lower() else c.expression[:45] + ("..." if len(c.expression) > 45 else ""))
                    did = f"{sid}_decision_{idx}"
                    lines.append(f'{did} [shape=diamond, label="{_dot_escape(label)}", fillcolor="#FADBD8"];')
                    lines.append(f'{sid} -> {did} [label="Constraint"];')
                    for t in targets:
                        lines.append(f'{did} -> {node_ids[t]} [label="Route when true", color=blue];')

        source_count = sum(1 for s in stages if stage_layer[s.name] == "source")
        join_count = sum(len(s.join_blocks) for s in stages)
        cols = sum(sum(1 for t in s.transformations if t.target_col != "Unknown") for s in stages)
        rules_count = sum(sum(1 for t in s.transformations if t.is_modified) for s in stages)
        decision_count = sum(len(s.constraints) for s in stages)
        summary = (
            "Job Summary:\n"
            f"- {source_count} Source Stages\n"
            f"- {join_count} Joins\n"
            f"- {cols} Columns Processed\n"
            f"- {rules_count} Business Rules Applied\n"
            f"- {decision_count} Constraint Routes"
        )
        lines.append(f'Note1 [shape=plaintext, label="{_dot_escape(summary)}", fontsize=8, fillcolor="white"];')
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
        high = self._build_arch_dot(graph_name, stages, links, include_lookup=False, include_decisions=False)
        detailed = self._build_arch_dot(graph_name + "_detailed", stages, links, include_lookup=True, include_decisions=True)
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
