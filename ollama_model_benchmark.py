#!/usr/bin/env python3

import argparse
import time
from datetime import datetime
import requests
import re
import threading

from ollama_api import *
from tasks import *
from utils import *
from prompts import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models for coding tasks."
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file (default: ollama_benchmark_results_<timestamp>.md)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Specific model names to benchmark (default: all installed)",
    )
    parser.add_argument(
        "--no-table-task",
        action="store_true",
        help="Skip the table task benchmark.",
    )
    parser.add_argument(
        "--no-coding-task",
        action="store_true",
        help="Skip the Python coding task benchmark.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Show model metadata for debugging.",
    )
    return parser.parse_args()

import concurrent.futures

def check_model_vram(ollama_url, model_name, max_retries=5, delay=1, result_container=None):
    print(f"## Checking VRAM for {model_name} (up to {max_retries} checks, each with 60s timeout)...")
    vram_usage_bytes = None
    for vram_attempt in range(1, max_retries + 1):
        print(
            f"    Main VRAM check attempt {vram_attempt}/{max_retries}...",
            end="",
            flush=True,
        )
        running_vram_map = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_running_models_vram, ollama_url)
            try:
                running_vram_map = future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                print(" VRAM check timed out after 60 seconds.")
                running_vram_map = None
            except Exception as e:
                print(f" VRAM check error: {e}")
                running_vram_map = None

        if running_vram_map and model_name in running_vram_map:
            vram_usage_bytes = running_vram_map.get(model_name)
            if vram_usage_bytes is not None:
                print(f" found. VRAM: {fmt_bytes(vram_usage_bytes)}")
                break
            else:
                print(f" listed, but 'size_vram' missing. Assuming 0 B.")
                vram_usage_bytes = 0
                break
        else:
            print(" model not yet listed in /api/ps.")
            if vram_attempt < max_retries:
                print(
                    f"    Will retry VRAM check in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                print(
                    f"    VRAM for {model_name} not found in /api/ps after all checks."
                )
    if vram_usage_bytes is None:
        print(f"    Final VRAM status for {model_name}: Not listed in /api/ps.")
    if result_container is not None:
        result_container["vram"] = vram_usage_bytes
    return vram_usage_bytes

def get_model_response(ollama_url, model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"[Error getting response for prompt '{prompt[:180]}...' from {model_name}: {e}]")
        return None

def main():
    args = parse_args()
    ollama_url = args.ollama_url
    selected_models = args.models
    run_table_task = not args.no_table_task
    run_coding_task = not args.no_coding_task
    verbose = args.verbose

    print("Listing installed Ollama models and their disk sizes...")
    try:
        models_details = list_models(ollama_url)
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return
    if not models_details:
        print("❌ No models found.")
        return

    model_names_on_disk = [md["name"] for md in models_details]
    # Filter out embed models (e.g., nomic-embed-text and any model with 'embed' in the name)
    def is_embed_model(name):
        return "embed" in name.lower()
    filtered_model_names_on_disk = [m for m in model_names_on_disk if not is_embed_model(m)]

    if selected_models:
        model_names = [m for m in filtered_model_names_on_disk if m in selected_models]
        if not model_names:
            print("❌ None of the specified models found (after filtering embed models).")
            return
    else:
        model_names = filtered_model_names_on_disk

    print(f"Found {len(model_names)} models (excluding embed models): {', '.join(model_names)}\n")
    results = []

    for model_detail in models_details:
        model_name = model_detail["name"]
        if model_name not in model_names:
            continue
        disk_size_bytes = model_detail["size"]
        print(
            f"\n# Testing model: {model_name} (Disk size: {fmt_bytes(disk_size_bytes)})"
        )
        # Fetch model metadata (e.g., max context window tokens)
        model_metadata = get_model_metadata(ollama_url, model_name)
        if args.metadata:
            print(f"[DEBUG] model_metadata for {model_name}: {repr(model_metadata)}")
        # Extract context_length_val from model metadata
        from utils import get_model_context_length
        
        context_length_val = get_model_context_length(ollama_url, model_name) or 8192
        
        time.sleep(1)

        # Start VRAM check in a background thread before loading model
        vram_result = {}
        vram_thread = threading.Thread(target=check_model_vram, args=(ollama_url, model_name, 5, 1, vram_result))
        vram_thread.start()

        print("## Loading model via /api/show...", end="", flush=True)
        load_time = warmup_model(ollama_url, model_name, SELF_IDENTIFY_PROMPT)

        if load_time is None:
            print(
                f" ❌ Skipping benchmark tests for {model_name} due to load error/timeout."
            )
            vram_thread.join()
            results.append(
                {
                    "model": model_name,
                    "load_time": None,
                    "disk_size": disk_size_bytes,
                    "vram_usage": None,
                    "table_gen_time": None,
                    "table_generated_tokens": None,
                    "table_tok_per_sec_eval": None,
                    "table_first_token_latency": None,
                    "table_output_length": None,
                    "table_task_match": False,
                    "table_task_reason": "Load failed",
                    "coding_gen_time": None,
                    "coding_generated_tokens": None,
                    "coding_tok_per_sec_eval": None,
                    "coding_first_token_latency": None,
                    "coding_output_length": None,
                    "coding_task_passed": False,
                    "coding_task_reason": "Load failed",
                }
            )
            continue
        print(f" done in {load_time:.3f}s.")

        vram_thread.join()
        vram_usage_bytes = vram_result.get("vram")

        # Query model for Self-ID, release date and developer/organization
        raw_self_id_response = get_model_response(ollama_url, model_name, SELF_IDENTIFY_PROMPT)
        raw_release_date_response = get_model_response(ollama_url, model_name, RELEASE_DATE_PROMPT)
        raw_developer_response = get_model_response(ollama_url, model_name, DEVELOPER_PROMPT)
        print(f"[DEBUG] Raw developer response for {model_name}: {repr(raw_developer_response)}")
        self_id_response = clean_metadata_response(raw_self_id_response)
        release_date_response = clean_metadata_response(raw_release_date_response)
        developer_response = clean_metadata_response(raw_developer_response)
        if not developer_response:
            print(f"[DEBUG] Developer name missing for {model_name}. Raw response: {repr(raw_developer_response)} | Metadata: {repr(model_metadata)}")

        # Measure max context using binary search
        print(f"Measuring max context for {model_name} (this may take a while)...")
        context_result = test_context_window(ollama_url, model_name, context_length_val)
        print(f"Measured max context for {model_name}: {context_result}")

        # Extract license, parameter_count, context_length, developer/company

        def get_model_field(meta, keys):
            """Recursively search for the first key found in meta or its subdicts."""
            if not isinstance(meta, dict):
                return None
            for key in keys:
                if key in meta:
                    return meta[key]
            # Search in subdicts
            for v in meta.values():
                if isinstance(v, dict):
                    found = get_model_field(v, keys)
                    if found is not None:
                        return found
            return None

        # --- Extraction for License, Params (K), Context Len ---

        # Prefer model_metadata['modelinfo'] for these fields if present
        modelinfo = model_metadata.get("modelinfo", {})

        # License: use modelinfo["general.license"] if available
        license_val = modelinfo.get("general.license", "")
        license_short = str(license_val)

        # Params (K): use modelinfo["general.parameter_count"] if available
        param_val = modelinfo.get("general.parameter_count")
        if param_val is not None:
            try:
                param_count_k = f"{int(param_val)//1000}K"
            except Exception:
                param_count_k = str(param_val) + "K"
        else:
            param_count_k = ""

        # Context Len: use modelinfo["context_length"] or any key ending with ".context_length"
        context_val = modelinfo.get("context_length")
        if context_val is None:
            for k, v in modelinfo.items():
                if k.endswith(".context_length"):
                    context_val = v
                    break
        context_length_val = str(context_val) if context_val is not None else ""

        # Remove developer_short, max_context_tokens, max_context_api_str, developer_meta, etc.

        current_model_results = {
            "model": clean_metadata_response(model_name),
            "self_id": self_id_response,
            "load_time": load_time,
            "disk_size": disk_size_bytes,
            "vram_usage": vram_usage_bytes,
            "license_short": license_short,
            "param_count_k": param_count_k,
            "context_length_val": context_length_val,
            "context_test": context_result,  # Changed from measured_max_context
            "release_date_response": release_date_response,
            "developer_response": developer_response,
            "table_task_reason": "Not run",
            "coding_task_reason": "Not run",
        }

        if run_table_task:
            print("\n## Generating Table response (streaming)...")
            (
                table_wall_time,
                table_output,
                table_gen_tokens,
                table_gen_eval_ns,
                table_ftl,
            ) = test_model_stream(ollama_url, model_name, TABLE_TEST_PROMPT)

            table_task_match, table_task_reason = (
                (False, "No output from model")
                if table_wall_time is None
                else check_table_and_languages(table_output)
            )

            current_model_results.update(
                {
                    "table_gen_time": table_wall_time,
                    "table_output_length": len(table_output) if table_output else 0,
                    "table_first_token_latency": table_ftl,
                    "table_generated_tokens": table_gen_tokens,
                    "table_tok_per_sec_eval": (table_gen_tokens / (table_gen_eval_ns / 1e9))
                    if table_gen_tokens and table_gen_eval_ns and table_gen_eval_ns > 0
                    else None,
                    "table_task_match": table_task_match,
                    "table_task_reason": table_task_reason,
                }
            )
            print(
                f"\n\n    Table Task: Gen time: {fmt(current_model_results['table_gen_time'])}s | Tok/s (eval): {fmt(current_model_results['table_tok_per_sec_eval'], 2)} | "
                f"Tokens: {fmt(current_model_results['table_generated_tokens'])} | First tok: {fmt(current_model_results['table_first_token_latency'])}s | "
                f"Output chars: {current_model_results['table_output_length']} | Match: {'✅' if current_model_results['table_task_match'] else '❌'}"
                f"{'' if current_model_results['table_task_match'] else ' Reason: ' + current_model_results['table_task_reason']}"
            )

        time.sleep(3)
        if run_coding_task:
            print("\n## Generating Python Coding response (streaming)...")
            (
                coding_wall_time,
                coding_output,
                coding_gen_tokens,
                coding_gen_eval_ns,
                coding_ftl,
            ) = test_model_stream(ollama_url, model_name, PYTHON_CODING_PROMPT)

            coding_task_passed, coding_task_reason = (
                (False, "No output from model")
                if coding_wall_time is None
                else check_python_coding_task(coding_output)
            )

            current_model_results.update(
                {
                    "coding_gen_time": coding_wall_time,
                    "coding_output_length": len(coding_output) if coding_output else 0,
                    "coding_first_token_latency": coding_ftl,
                    "coding_generated_tokens": coding_gen_tokens,
                    "coding_tok_per_sec_eval": (
                        coding_gen_tokens / (coding_gen_eval_ns / 1e9)
                    )
                    if coding_gen_tokens and coding_gen_eval_ns and coding_gen_eval_ns > 0
                    else None,
                    "coding_task_passed": coding_task_passed,
                    "coding_task_reason": coding_task_reason,
                }
            )
            print(
                f"\n    Coding Task: Gen time: {fmt(current_model_results['coding_gen_time'])}s | Tok/s (eval): {fmt(current_model_results['coding_tok_per_sec_eval'], 2)} | "
                f"Tokens: {fmt(current_model_results['coding_generated_tokens'])} | First tok: {fmt(current_model_results['coding_first_token_latency'])}s | "
                f"Output chars: {current_model_results['coding_output_length']} | Passed: {'✅' if current_model_results['coding_task_passed'] else '❌'}"
                f"{'' if current_model_results['coding_task_passed'] else ' Reason: ' + current_model_results['coding_task_reason']}"
            )
        results.append(current_model_results)
        unload_model(ollama_url, model_name)

    def sort_key(res):
        coding_passed_score = (
            -int(res.get("coding_task_passed", False))
            if res.get("coding_task_passed") is not None
            else 1
        )
        table_match_score = (
            -int(res.get("table_task_match", False)) if res.get("table_task_match") is not None else 1
        )
        # Context Len (descending, so negative for bigger first)
        try:
            context_len = -int(res.get("context_length_val", 0))
        except Exception:
            context_len = 0
        coding_tok_s_eval = res.get("coding_tok_per_sec_eval")
        sortable_tok_s = -(
            coding_tok_s_eval if coding_tok_s_eval is not None else -float("inf")
        )
        vram_usage_val = (
            res.get("vram_usage") if res.get("vram_usage") is not None else float("inf")
        )
        return (
            coding_passed_score,
            table_match_score,
            context_len,
            sortable_tok_s,
            vram_usage_val,
            res["model"].lower(),
        )

    results.sort(key=sort_key)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create results directory
    import os
    os.makedirs("results", exist_ok=True)
    
    # Update markdown filename to use results folder
    md_filename = args.output or f"results/ollama_benchmark_results_{now}.md"
    
    md_lines = [f"# Ollama Model Benchmark Results ({now})\n"]
    lang_list_str = ", ".join(LANGUAGES[:-1]) + ", and " + LANGUAGES[-1]
    # Print all prompts used in tests
    md_lines.append("## Prompts Used in Tests\n")
    md_lines.append("```")
    md_lines.append(f"TABLE_TEST_PROMPT:\n{TABLE_TEST_PROMPT}\n")
    md_lines.append(f"PYTHON_CODING_PROMPT:\n{PYTHON_CODING_PROMPT}\n")
    md_lines.append(f"SELF_IDENTIFY_PROMPT:\n{SELF_IDENTIFY_PROMPT}\n")
    md_lines.append(f"RELEASE_DATE_PROMPT:\n{RELEASE_DATE_PROMPT}\n")
    md_lines.append(f"DEVELOPER_PROMPT:\n{DEVELOPER_PROMPT}\n")
    md_lines.append("```\n")

    md_lines.append(
        "| Model | Self-ID (resp) | Disk Size | VRAM Usage | License (metadata) | Developer (resp) | Params (K) | Context Len | Release Date | Load (s) | Table Gen (s) | Table Tok/s (eval) | Table FT (s) | Code Gen (s) | Code Tok/s (eval) | Code FT (s) | Context Test | Table Test | Coding Test |"
        
    )
    md_lines.append(
        "|-------|:-------:|:---------:|:----------:|:-------:|:----------------:|:----------:|:-----------:|:----------------------:|:------------:|:--------:|:-------------:|:------------------:|:------------:|:------------:|:-----------------:|:-----------:|:-----------:|:-------------:|"
    )
    for r in results:
        md_lines.append(
            f"| `{r['model']}` | `{r.get('self_id', '')}` | {fmt_bytes(r['disk_size'])} | {fmt_bytes(r['vram_usage'])} | {r.get('license_short', '')} | {r.get('developer_response', '')} | {r.get('param_count_k', '')} | {r.get('context_length_val', '')} | {r.get('release_date_response', '')} | {fmt(r['load_time'])} | "
            f"{fmt(r.get('table_gen_time'))} | {fmt(r.get('table_tok_per_sec_eval'), 2)} | {fmt(r.get('table_first_token_latency'))} | "
            f"{fmt(r.get('coding_gen_time'))} | {fmt(r.get('coding_tok_per_sec_eval'), 2)} | {fmt(r.get('coding_first_token_latency'))} | "
            f"{r.get('context_test', '')} | {'✅' if r.get('table_task_match') else '❌'} | {'✅' if r.get('coding_task_passed') else '❌'} |"
        )
    md_output = "\n".join(md_lines)
    print("\nMarkdown summary:\n")
    print(md_output)
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(md_output)
    print(f"\nSaved results to {md_filename}")

if __name__ == "__main__":
    main()
