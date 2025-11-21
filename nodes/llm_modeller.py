import os
import json
import re
from openai import OpenAI
from typing import Dict, Any

def call_openai_and_parse(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 5000) -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=key)

    # Call the OpenAI Chat Completions API (new v1 syntax)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=max_tokens
    )

    text = resp.choices[0].message.content

    # Extract the first JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        json_text = m.group(0)
    else:
        json_text = text  # fallback: try the whole response

    # Attempt to parse JSON
    try:
        parsed = json.loads(json_text)
        return parsed
    except Exception as e:
        # Save raw output for debugging
        os.makedirs("output", exist_ok=True)
        with open("output/llm_raw.txt", "w", encoding="utf-8") as f:
            f.write(text)
        raise RuntimeError(
            f"Failed to parse JSON from LLM: {e}.\n"
            f"Raw output saved to output/llm_raw.txt"
        )
