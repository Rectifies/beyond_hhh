"""
Collect model outputs for each prompt.

Sends prompts to model APIs and saves responses as JSON files.

Usage:
    python scripts/collect_outputs.py                          # All models, all prompts
    python scripts/collect_outputs.py --model gpt-4o           # Single model
    python scripts/collect_outputs.py --prompt P1              # Single prompt
    python scripts/collect_outputs.py --run 2                  # Run 2 of N (for repeated sampling)
    python scripts/collect_outputs.py --run 2 --skip-controls  # Skip C1-C5 on repeat runs
"""

import argparse
import json
import os
from datetime import datetime

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    MODELS,
    DATA_DIR,
)
from prompts import PROMPTS


def call_openai(model_id: str, prompt: str) -> dict:
    """Send prompt to OpenAI API and return response."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return {
        "output": response.choices[0].message.content,
        "model_used": response.model,
    }


def call_anthropic(model_id: str, prompt: str) -> dict:
    """Send prompt to Anthropic API and return response."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "output": response.content[0].text,
        "model_used": response.model,
    }


def call_google(model_id: str, prompt: str) -> dict:
    """Send prompt to Google Gemini API and return response."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=1.0),
    )
    return {
        "output": response.text,
        "model_used": model_id,
    }


PROVIDER_FUNCTIONS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


def collect_output(
    prompt_id: str, prompt_data: dict, model_name: str, model_config: dict
) -> dict:
    """Collect output from a model for a given prompt."""
    record = {
        "prompt_id": prompt_id,
        "prompt_name": prompt_data["name"],
        "prompt_text": prompt_data["prompt"],
        "platform": prompt_data["platform"],
        "social_goal": prompt_data["social_goal"],
        "model": model_name,
        "model_id": model_config["model_id"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "interface": "api",
    }

    provider = model_config["provider"]
    call_fn = PROVIDER_FUNCTIONS[provider]

    try:
        result = call_fn(model_config["model_id"], prompt_data["prompt"])
        record.update(result)
        print(f"  Collected output ({len(result['output'])} chars)")
    except Exception as e:
        print(f"  API error: {e}")
        record["output"] = f"API ERROR: {e}"

    return record


def save_record(record: dict, run: int = 1) -> str:
    """Save a single output record to data/raw/.

    Run 1 files have no suffix (preserves compatibility with existing data).
    Run 2+ files are saved as {prompt}_{model}_run{N}.json.
    """
    output_dir = os.path.join(DATA_DIR, "raw")
    run_suffix = f"_run{run}" if run > 1 else ""
    filename = f"{record['prompt_id']}_{record['model'].replace(' ', '_')}{run_suffix}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Collect model outputs")
    parser.add_argument("--model", help="Test single model")
    parser.add_argument("--prompt", help="Test single prompt (e.g., P1)")
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number for repeated sampling (default: 1). "
             "Run 1 preserves original filenames. Run 2+ adds _run{N} suffix.",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip control prompts (C1-C5). Useful for repeat runs where "
             "control behaviour is already established.",
    )
    args = parser.parse_args()

    models_to_test = MODELS
    if args.model:
        models_to_test = {
            k: v for k, v in MODELS.items() if k == args.model
        }

    prompts_to_test = PROMPTS
    if args.prompt:
        prompts_to_test = {
            k: v for k, v in PROMPTS.items() if k == args.prompt
        }
    if args.skip_controls:
        prompts_to_test = {
            k: v for k, v in prompts_to_test.items()
            if not k.startswith("C")
        }

    total = len(models_to_test) * len(prompts_to_test)
    run_label = f" (run {args.run})" if args.run > 1 else ""
    print(f"Collecting outputs{run_label}: {len(prompts_to_test)} prompts x "
          f"{len(models_to_test)} models = {total} combinations")
    print(f"Temperature: 1.0 (all providers)\n")

    for prompt_id, prompt_data in prompts_to_test.items():
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_id}: {prompt_data['name']}")
        print(f"{'='*60}")

        for model_name, model_config in models_to_test.items():
            print(f"\n  Model: {model_name}")

            record = collect_output(
                prompt_id, prompt_data, model_name, model_config
            )

            record["run"] = args.run
            filepath = save_record(record, run=args.run)
            print(f"  Saved: {filepath}")

    print(f"\n\nDone. {total} files saved to {DATA_DIR}/raw/")


if __name__ == "__main__":
    main()
