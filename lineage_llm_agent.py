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
    build_detailed_dot,
    build_global_links,
    build_high_level_dot,
    compute_complexity,
    detect_undefined_references,
    longest_chain,
    parse_stages,
    sanitize_id,
)


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

    def generate_diagram(self, state: Dict[str, Any]) -> None:
        if state.get("final_output"):
            return
        stages: List[Stage] = state["stages"]
        links = state["links"]
        graph_name = sanitize_id(state["file_stem"], "lineage")
        chain_len = longest_chain([(a, b) for a, b, _ in links])
        rankdir = "TB" if (len(stages) > 10 or chain_len > 8) else "LR"

        high = build_high_level_dot(graph_name, stages, rankdir, links)
        detailed = build_detailed_dot(graph_name, stages, rankdir, links, state.get("transformed_columns", 0))
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
