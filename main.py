#!/usr/bin/env python3
"""
main.py – now supporting MULTIPLE input files.
"""

import os
import argparse
from pathlib import Path
import json

from nodes.file_loader import load_file
from nodes.prompt_builder import build_prompt
from nodes.llm_modeller import call_openai_and_parse
from nodes.output_writer import write_outputs

def ensure_output_dir(path="output"):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def run(input_paths, output_dir):
    ensure_output_dir(output_dir)

    print("\n=== MULTI-FILE DATA MODELLER ===\n")

    # -------------------------------
    # 1. Load ALL input files
    # -------------------------------
    all_contents = []
    merged_columns = []
    merged_samples = []

    print("[1] Loading files:")

    for path in input_paths:
        print(f"    → {path}")
        file_data = load_file(path, sample_rows=50)

        all_contents.append(file_data)
        merged_columns.extend(file_data.get("columns", []))
        merged_samples.extend(file_data.get("sample_rows", []))

    # Deduplicate columns
    merged_columns = list(dict.fromkeys(merged_columns))

    # Limit sample rows
    merged_samples = merged_samples[:50]

    content = {
        "columns": merged_columns,
        "sample_rows": merged_samples,
        "source_type": "multi_file"
    }

    print(f"\n    Loaded {len(input_paths)} files")
    print(f"    Total merged columns: {len(merged_columns)}")
    print(f"    Total sample rows: {len(merged_samples)}\n")

    # -------------------------------
    # 2. Build prompt for ALL files
    # -------------------------------
    file_list_string = ", ".join(input_paths)
    prompt = build_prompt(file_list_string, content)

    # -------------------------------
    # 3. Call OpenAI LLM
    # -------------------------------
    print("[3] Calling OpenAI for semantic modelling...\n")
    parsed = call_openai_and_parse(prompt)

    # -------------------------------
    # 4. Write Outputs
    # -------------------------------
    print("[4] Writing outputs...\n")
    write_outputs(parsed, output_dir)

    print("✔ DONE!")
    print(f"Outputs saved to: {output_dir}")
    print("  - erd_openai.json")
    print("  - ddl_openai.sql")
    print("  - validation_openai.json")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        nargs="+",                      # ← THIS enables multi-file support
        required=True,
        help="One or more input files (CSV, Excel, JSON)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory"
    )
    args = parser.parse_args()

    run(args.input, args.output)
