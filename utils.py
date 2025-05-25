import re
import ollama

def fmt_bytes(byte_val):
    if byte_val is None:
        return "-"
    if not isinstance(byte_val, (int, float)) or byte_val < 0:
        return "-"
    if byte_val == 0:
        return "0 B"
    gb = byte_val / (1024 ** 3)
    mb = byte_val / (1024 ** 2)
    kb = byte_val / 1024
    if gb >= 0.1:
        return f"{gb:.2f} GB"
    if mb >= 0.1:
        return f"{mb:.2f} MB"
    if kb >= 1:
        return f"{kb:.0f} KB"
    return f"{byte_val} B"

def fmt(val, precision=3):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)

def clean_metadata_response(text):
    if not text:
        return ""
    # Remove <think>...</think> blocks (non-greedy, multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove newlines
    text = text.replace("\n", "").replace("\r", "")
    return text.strip()

def get_model_context_length(ollama_url, model_name):
    """
    Get a model's context length from its metadata.
    
    Args:
        ollama_url (str): URL of the Ollama API
        model_name (str): Name of the model
        
    Returns:
        int or None: The context length if found, None otherwise
    """
    try:
        client = ollama.Client(host=ollama_url)
        model_details = client.show(model_name)
        
        # Look for any key ending with 'context_length'
        for key in model_details.modelinfo:
            if key.endswith('.context_length'):
                return int(model_details.modelinfo[key])
        
        return None
        
    except Exception as e:
        print(f"Error getting context length for {model_name}: {e}")
        return None
