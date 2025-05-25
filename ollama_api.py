import time
import json
from typing import List, Dict, Optional, Tuple
import ollama

OLLAMA_HOST_DEFAULT = "http://10.10.10.110:11434"

def get_ollama_client(ollama_url: Optional[str] = None):
    return ollama.Client(host=ollama_url or OLLAMA_HOST_DEFAULT)

def list_models(ollama_url: Optional[str] = None) -> List[Dict]:
    """Fetches all models and their disk sizes using the ollama client."""
    client = get_ollama_client(ollama_url)
    models_data = client.list().get("models", [])
    result = []
    for model in models_data:
        # Support both dict and object (e.g., namedtuple/dataclass)
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            size = model.get("size")
            if not name:
                print(f"[DEBUG] Skipping dict model entry without 'name' or 'model': {repr(model)}")
                continue
        elif hasattr(model, "model") and hasattr(model, "size"):
            name = getattr(model, "model")
            size = getattr(model, "size")
        else:
            print(f"[DEBUG] Skipping unknown model entry: {repr(model)}")
            continue
        result.append({"name": name, "size": size})
    return result

def warmup_model(ollama_url: Optional[str], model_name: str, self_identify_prompt: str) -> Optional[float]:
    """
    Ensures the model is loaded using ollama client .show(), then asks the model to identify itself
    via a quick .generate() call. Returns the total time taken for both operations.
    """
    client = get_ollama_client(ollama_url)
    start_time = time.time()
    try:
        print(f"    Initiating /api/show for {model_name}...", end="", flush=True)
        # .show() loads the model and returns metadata
        client.show(model_name)
        print(f" /api/show OK.", end="")

        print(f" Asking model to self-identify...", end="", flush=True)
        generate_response_data = client.generate(
            model=model_name,
            prompt=self_identify_prompt,
            stream=False,
        )
        model_self_name = generate_response_data.get("response", "").strip()

        if model_self_name:
            max_len_self_name = 70
            display_name = (
                model_self_name
                if len(model_self_name) <= max_len_self_name
                else model_self_name[: max_len_self_name - 3] + "..."
            )
            print(f" Self-ID: '{display_name}'.", end="")
        else:
            print(f" No self-ID response or empty.", end="")

    except Exception as e:
        current_op_duration = time.time() - start_time
        print(
            f"\n    [Error during warmup for {model_name} (duration: {current_op_duration:.2f}s): {e}]"
        )
        return None

    elapsed_total = time.time() - start_time
    return elapsed_total

def get_running_models_vram(ollama_url: Optional[str] = None) -> Dict[str, int]:
    """Fetches VRAM usage for currently running models using the ollama client."""
    try:
        client = get_ollama_client(ollama_url)
        running_models_data = client.ps().get("models", [])
        vram_map = {
            model.get("name"): model.get("size_vram", 0)
            for model in running_models_data
            if model.get("name")
        }
        return vram_map
    except Exception as e:
        print(f"\n    [Error fetching running models with ollama client: {e}]")
        return {}

def test_model_stream(
    ollama_url: Optional[str], model_name: str, prompt: str
) -> Tuple[Optional[float], str, Optional[int], Optional[int], Optional[float]]:
    """
    Streams model output using the ollama client, measuring timing and collecting stats.
    """
    client = get_ollama_client(ollama_url)
    print("    Output:\n\n", end="", flush=True)
    response_text = ""
    first_token_latency = None
    start_time = time.time()
    first_token_time = None
    generated_tokens_count = None
    gen_eval_duration_ns = None

    try:
        for data in client.generate(model=model_name, prompt=prompt, stream=True):
            chunk = data.get("response", "")
            if first_token_time is None and chunk:
                first_token_time = time.time()
                first_token_latency = first_token_time - start_time

            print(chunk, end="", flush=True)
            response_text += chunk

            if data.get("done"):
                generated_tokens_count = data.get("eval_count")
                gen_eval_duration_ns = data.get("eval_duration")
                break
        wall_time = time.time() - start_time
    except Exception as e:
        print(f"\n    [Error during generation: {e}]")
        return None, "", None, None, None

    print()
    return (
        wall_time,
        response_text,
        generated_tokens_count,
        gen_eval_duration_ns,
        first_token_latency,
    )

def get_model_metadata(ollama_url: Optional[str], model_name: str) -> dict:
    """
    Fetches model metadata using the ollama client .show() for the given model.
    Returns the full JSON response, or an empty dict on error.
    """
    try:
        client = get_ollama_client(ollama_url)
        return client.show(model_name)
    except Exception as e:
        print(f"[Error fetching metadata for {model_name} with ollama client: {e}]")
        return {}

def unload_model(ollama_url: Optional[str], model_name: str):
    """
    Unloads a model using the ollama client by sending a generate with keep_alive=0.
    """
    try:
        client = get_ollama_client(ollama_url)
        # The prompt can be empty or a space, as in the original
        client.generate(model=model_name, prompt=" ", keep_alive=0, stream=False)
        print(f"    Sent unload signal for model: {model_name}")
    except Exception as e:
        print(f"❌    [Failed to send unload signal for model {model_name}: {e}]")
