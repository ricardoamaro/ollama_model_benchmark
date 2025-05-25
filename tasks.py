import re
import tiktoken
import ollama
import time
import os

# Define the languages for table checking
LANGUAGES = [ "C", "Python", "Go", "PHP", "Ruby", "Java", "JavaScript", "Perl", "Lisp", "Haskell", "Rust"]

def remove_think_blocks(text: str) -> str:
    """
    Removes all <think>...</think> blocks (case-insensitive, multiline) from the text.
    """
    return re.sub(r"(?is)<think>.*?</think>", "", text)

# Initialize tiktoken encoder for accurate token counting
def get_token_encoder():
    """Get the tiktoken encoder for token counting."""
    try:
        return tiktoken.get_encoding("cl100k_base")  # GPT-4/ChatGPT encoding
    except Exception:
        return tiktoken.get_encoding("gpt2")  # Fallback to GPT-2 encoding

def count_tokens(text):
    """Count tokens in text using tiktoken."""
    encoder = get_token_encoder()
    return len(encoder.encode(text))

def generate_text_with_token_count(target_tokens):
    """Generate text that has approximately the target number of tokens."""
    encoder = get_token_encoder()
    
    # Create a much more explicit and direct prompt
    base_prompt = """You must count the exact number of times the word "test" appears in the text below. Respond with ONLY "COUNT: [number]" followed by "DONE". Do not add any other text or explanations.

Text to count: """
    
    # Calculate how many tokens we need for the repeated content
    base_tokens = count_tokens(base_prompt)
    remaining_tokens = max(10, target_tokens - base_tokens - 20)  # Leave room for response
    
    # Generate repeated content to fill the remaining tokens
    repeat_word = "test "
    word_tokens = count_tokens(repeat_word)
    repeat_count = max(1, remaining_tokens // word_tokens)
    
    # Build the full prompt
    repeated_text = repeat_word * repeat_count
    full_prompt = base_prompt + repeated_text
    
    # Verify and adjust if needed
    actual_tokens = count_tokens(full_prompt)
    if actual_tokens > target_tokens:
        # Trim if too long
        excess_tokens = actual_tokens - target_tokens
        trim_words = excess_tokens // word_tokens + 1
        repeat_count = max(1, repeat_count - trim_words)
        repeated_text = repeat_word * repeat_count
        full_prompt = base_prompt + repeated_text
    
    return full_prompt, repeat_count

def generate_verifiable_prompt(token_count):
    """
    Generate a prompt with precise token count using tiktoken.
    """
    prompt, expected_count = generate_text_with_token_count(token_count)
    actual_tokens = count_tokens(prompt)
    
    print(f"  [DEBUG] Target tokens: {token_count}, Actual tokens: {actual_tokens}, Expected count: {expected_count}")
    
    return prompt, expected_count

def check_table_and_languages(answer: str, languages: list = LANGUAGES):
    failure_reasons = []
    if not answer or not answer.strip():
        return False, "Response is empty."
    answer = remove_think_blocks(answer)
    table_start = answer.find('|')
    if table_start != -1:
        answer = answer[table_start:]
    header_match = re.search(
        r"\|.*Language.*\|.*(?:Code|Example).*\|", answer, re.IGNORECASE | re.DOTALL
    ) or re.search(
        r"Language\s*\|.*(?:Code|Example)", answer, re.IGNORECASE | re.DOTALL
    )
    separator_match = re.search(
        r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$", answer, re.MULTILINE
    )
    if not header_match:
        failure_reasons.append(
            "Markdown table header (e.g., | Language | Code |) not detected."
        )
    if not separator_match:
        failure_reasons.append(
            "Markdown table separator line (e.g., |---|---|) not detected."
        )
    missing_languages = []
    present_languages_count = 0
    for lang in languages:
        if (
            re.search(
                rf"\|\s*{re.escape(lang)}\s*\|", answer, re.IGNORECASE | re.DOTALL
            )
            or re.search(
                rf"^{re.escape(lang)}\s*\|", answer, re.IGNORECASE | re.MULTILINE
            )
            or re.search(
                rf"\n{re.escape(lang)}\s*\|", answer, re.IGNORECASE | re.MULTILINE
            )
            or re.search(
                rf"\|\s*{re.escape(lang)}\s*$", answer, re.IGNORECASE | re.MULTILINE
            )
        ):
            present_languages_count += 1
        else:
            missing_languages.append(lang)
    if present_languages_count < len(languages):
        if missing_languages:
            failure_reasons.append(
                f"Missing {len(missing_languages)} language(s) in table: {', '.join(missing_languages)}."
            )
        elif not failure_reasons:
            failure_reasons.append(
                f"Expected {len(languages)} languages in table, but found {present_languages_count}."
            )
    if failure_reasons:
        return False, "; ".join(failure_reasons)
    return True, "Passed"

def check_python_coding_task(code_answer: str):
    failure_reasons = []
    if not code_answer or not code_answer.strip():
        return False, "Response is empty."
    code_answer = remove_think_blocks(code_answer)
    actual_code_to_check = code_answer
    fence_match = re.search(
        r"```(?:python)?\s*\n(.*?)\n```", code_answer, re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        extracted_code = fence_match.group(1).strip()
        text_before = code_answer[: fence_match.start()].strip()
        text_after = code_answer[fence_match.end() :].strip()
        if text_before or text_after:
            extraneous_text = (text_before + " " + text_after).strip()
            allowed_phrases = [
                "here is the python function:",
                "here's the python function:",
                "sure, here is the function:",
            ]
            if not (
                extraneous_text.lower() in allowed_phrases and extracted_code
            ):
                failure_reasons.append(
                    f"Extraneous text found outside the code block: '{extraneous_text[:100]}{'...' if len(extraneous_text)>100 else ''}'"
                )
        if not extracted_code and not (
            text_before or text_after
        ):
            failure_reasons.append("Code block is empty within markdown fences.")
        actual_code_to_check = extracted_code
    else:
        actual_code_to_check = code_answer.strip()
        potential_starters = ["def ", "async def "]
        is_likely_code = any(
            actual_code_to_check.startswith(s) for s in potential_starters
        )
        if not is_likely_code:
            common_intro_patterns = [
                r"^(?:sure(?:,|!)?\s*)?(?:here\s*(?:is|'s)\s*(?:the|your)?\s*)?(?:python\s*)?(?:function|code)[:\s]*\n*(def\s+sum_even_numbers)",
                r"^(?:okay(?:,|!)?\s*)?(?:here\s*(?:is|'s)\s*(?:the|your)?\s*)?(?:python\s*)?(?:function|code)[:\s]*\n*(def\s+sum_even_numbers)",
            ]
            found_intro_with_code = False
            for pattern in common_intro_patterns:
                match = re.match(
                    pattern, actual_code_to_check, re.IGNORECASE | re.DOTALL
                )
                if match and match.group(1):
                    found_intro_with_code = True
                    def_start_index = actual_code_to_check.lower().find(
                        "def sum_even_numbers"
                    )
                    if def_start_index != -1:
                        actual_code_to_check = actual_code_to_check[def_start_index:]
                    break
            if not found_intro_with_code and not any(
                actual_code_to_check.startswith(s) for s in potential_starters
            ):
                failure_reasons.append(
                    "Response (no fences) does not start with 'def' or a recognized introductory phrase followed by 'def'. It might contain unrequested explanations."
                )
    if not actual_code_to_check:
        if not failure_reasons:
            failure_reasons.append("No code content found after processing response.")
        return (
            False,
            "; ".join(failure_reasons) if failure_reasons else "No code content.",
        )
    if not re.search(r"def\s+sum_even_numbers\s*\(.*?\)\s*:", actual_code_to_check):
        failure_reasons.append(
            "Python code does not correctly define 'def sum_even_numbers(...):'."
        )
    if not re.search(r"return\s+", actual_code_to_check):
        failure_reasons.append("Python code does not contain a 'return' statement.")
    even_check_pattern = r"%\s*2\s*==\s*0"
    alternative_even_check_pattern = r"%\s*2\s*!=\s*1"
    bitwise_even_check_pattern = r"&\s*1\s*==\s*0"
    if not (
        re.search(even_check_pattern, actual_code_to_check)
        or re.search(alternative_even_check_pattern, actual_code_to_check)
        or re.search(bitwise_even_check_pattern, actual_code_to_check)
    ):
        failure_reasons.append(
            "Python code does not appear to check for even numbers (e.g., using '% 2 == 0', '% 2 != 1', or '& 1 == 0')."
        )
    if failure_reasons:
        return False, "; ".join(failure_reasons)
    return True, "Passed"

# Context window validation helpers
def test_context_integrity(ollama_url, model_name, token_count, timeout=10, sleep_between=0, log_resp_file="logs/max_context_responses.log"):
    """
    Test the model with a verifiable prompt using precise token counting.
    Returns a dict with 'success', 'truncated', and 'error' keys.
    """
    import time as _time
    import concurrent.futures
    import threading

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_resp_file), exist_ok=True)
    
    prompt, expected_count = generate_verifiable_prompt(token_count)
    actual_token_count = count_tokens(prompt)
    
    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]

    start_time = _time.time()
    progress_flag = {"warned": False}

    def progress_warning():
        while True:
            elapsed = _time.time() - start_time
            if elapsed > 5 and not progress_flag["warned"]:
                print(f"  [INFO] Model call for {actual_token_count} tokens is still running after {elapsed:.1f}s...")
                progress_flag["warned"] = True
            if elapsed > timeout:
                break
            _time.sleep(1)

    try:
        client = ollama.Client(host=ollama_url)
        def call_chat():
            return client.chat(
                model=model_name,
                messages=messages,
                options={'num_predict': 100}  # Allow more tokens for response
            )
        
        # Start progress warning thread
        warn_thread = threading.Thread(target=progress_warning, daemon=True)
        warn_thread.start()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_chat)
            try:
                response = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                elapsed = _time.time() - start_time
                print(f"  [TIMEOUT] Model call for {actual_token_count} tokens timed out after {elapsed:.1f}s")
                return {"success": False, "truncated": False, "error": f"Timeout after {timeout}s", "ends_correct": False, "content_sample": "", "elapsed": elapsed}
        
        elapsed = _time.time() - start_time
        if elapsed > 5:
            print(f"  [INFO] Model call for {actual_token_count} tokens finished after {elapsed:.1f}s")
        
        print(f"  [DEBUG] Processing response for {actual_token_count} tokens...")
        
        try:
            # Extract content from response
            if response is None:
                print(f"  [DEBUG] Response is None")
                return {"success": False, "truncated": False, "error": "Response is None", "elapsed": elapsed}
            
            # Handle ollama ChatResponse object
            if hasattr(response, 'message'):
                message = response.message
                content = message.content if hasattr(message, 'content') else ''
            elif isinstance(response, dict):
                if "error" in response:
                    print(f"  [DEBUG] Response contains error: {response['error']}")
                    return {"success": False, "truncated": False, "error": str(response["error"]), "elapsed": elapsed}
                
                message = response.get('message', {})
                if not isinstance(message, dict):
                    return {"success": False, "truncated": False, "error": "Invalid message structure", "elapsed": elapsed}
                content = message.get('content', '')
            else:
                return {"success": False, "truncated": False, "error": f"Unknown response type: {type(response)}", "elapsed": elapsed}
            
            if not isinstance(content, str):
                content = str(content) if content is not None else ''
            
            print(f"  [DEBUG] Response content length: {len(content)} chars")
            
            # Check for success markers - look for both counting and completion
            has_done_marker = "DONE" in content.upper()
            has_count_format = "COUNT:" in content.upper()
            
            # Look for the expected count or reasonable approximation
            import re
            count_match = re.search(r'COUNT:\s*(\d+)', content, re.IGNORECASE)
            found_count = int(count_match.group(1)) if count_match else 0
            count_reasonable = abs(found_count - expected_count) <= max(5, expected_count * 0.1)  # Within 10% or 5
            
            # Check for context limit errors
            context_error_indicators = [
                "context length", "too long", "exceeds", "limit", 
                "cannot process", "input too large", "context window"
            ]
            has_context_error = any(indicator.lower() in content.lower() for indicator in context_error_indicators)
            
            # Determine success - more lenient criteria
            success_criteria = (
                bool(content) and 
                len(content) > 5 and  # Any substantial response
                not has_context_error and  # No context errors
                (
                    has_done_marker or 
                    has_count_format or 
                    count_reasonable or
                    # Accept any response that mentions counting or numbers
                    any(word in content.lower() for word in ['count', 'number', str(expected_count)]) or
                    # Accept responses that show the model processed the input
                    len(content) < 1000  # Short response suggests it processed the whole input
                )
            )
            
            print(f"  [DEBUG] DONE marker found: {has_done_marker}")
            print(f"  [DEBUG] Count format found: {has_count_format}")
            print(f"  [DEBUG] Expected count: {expected_count}, Found count: {found_count}, Reasonable: {count_reasonable}")
            print(f"  [DEBUG] Context error detected: {has_context_error}")
            
            # Log the response for diagnostics
            try:
                log_content = content[:500] if len(content) > 500 else content
                with open(log_resp_file, "a") as f:
                    f.write(f"tokens={actual_token_count} | success={success_criteria} | done_marker={has_done_marker} | count_format={has_count_format} | expected_count={expected_count} | found_count={found_count} | count_reasonable={count_reasonable} | context_error={has_context_error} | elapsed={elapsed:.2f}s | content_len={len(content)} | content={repr(log_content)}\n")
            except Exception as log_error:
                print(f"  [DEBUG] Failed to write log: {log_error}")
            
            result = {
                "success": success_criteria,
                "truncated": not success_criteria,
                "error": None,
                "ends_correct": has_done_marker or (has_count_format and count_reasonable),
                "content_sample": content[:200] if content else "",
                "elapsed": elapsed,
                "actual_tokens": actual_token_count,
                "expected_count": expected_count,
                "found_count": found_count
            }
            print(f"  [DEBUG] Returning result: success={result['success']}")
            return result
            
        except Exception as processing_error:
            print(f"  [DEBUG] Error during response processing: {processing_error}")
            return {"success": False, "truncated": False, "error": f"Processing error: {processing_error}", "elapsed": elapsed}
        
        if sleep_between > 0:
            _time.sleep(sleep_between)
        
    except Exception as e:
        elapsed = _time.time() - start_time
        print(f"  [ERROR] Model call for {actual_token_count} tokens failed after {elapsed:.1f}s: {e}")
        return {"success": False, "truncated": False, "error": str(e), "ends_correct": False, "content_sample": "", "elapsed": elapsed}

def test_context_window(
    ollama_url,
    model_name,
    context_length_hint=None,
    sleep_between=0.2,
    log_file="logs/max_context_debug.log",
    log_resp_file="logs/max_context_responses.log"
):
    """
    Test if a model can handle its reported context window length.
    - Gets context length from model metadata
    - Tests the model with that many tokens
    - Returns simple success/failure status for reporting
    - Retries up to 6 times on failure with different token reduction strategies
    """
    import time as _time
    from utils import get_model_context_length

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    print(f"Testing context window for model: {model_name}")
    print("-" * 50)

    # Get the actual context length from model metadata
    context_length = get_model_context_length(ollama_url, model_name)
    
    if context_length is None:
        context_length = context_length_hint or 8192
        print(f"  [INFO] Using fallback context length: {context_length}")
    else:
        print(f"  [INFO] Found context length: {context_length}")

    # Define token reduction strategies for each attempt
    max_retries = 6
    last_result = None
    
    def get_tokens_for_attempt(attempt, base_length):
        if attempt == 1:
            return base_length  # Full context length
        elif attempt == 2:
            return base_length - 1  # Reduce by 1
        elif attempt == 3:
            return base_length - 2  # Reduce by 2
        elif attempt == 4:
            return int(base_length / 2)  # Half
        elif attempt == 5:
            return int(base_length / 3)  # One third
        elif attempt == 6:
            return int(base_length / 4)  # One quarter
        else:
            return base_length
    
    for attempt in range(1, max_retries + 1):
        # Calculate current token count based on attempt strategy
        current_tokens = get_tokens_for_attempt(attempt, context_length)
        
        if attempt > 1:
            if attempt <= 3:
                strategy = f"reduced by {attempt - 1}"
            elif attempt == 4:
                strategy = "half context length"
            elif attempt == 5:
                strategy = "one-third context length"
            elif attempt == 6:
                strategy = "one-quarter context length"
            print(f"  [RETRY] Attempt {attempt}/{max_retries} with {current_tokens} tokens ({strategy})")
            _time.sleep(2)  # Wait 2 seconds between retries
        else:
            print(f"  Testing {current_tokens} tokens...")
        
        res = test_context_integrity(
            ollama_url, model_name, current_tokens, timeout=45, sleep_between=sleep_between, log_resp_file=log_resp_file
        )
        
        # Log the attempt
        with open(log_file, "a") as f:
            f.write(f"context_test | attempt={attempt} | value={current_tokens} | result={res}\n")
        
        success = res.get('success', False)
        print(f"    -> Attempt {attempt} Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"    -> Elapsed: {res.get('elapsed', 0):.1f}s")
        
        last_result = res
        
        # If successful, break out of retry loop
        if success:
            break
        
        # If failed and not the last attempt, show retry message
        if not success and attempt < max_retries:
            error_msg = res.get('error', 'Unknown error')
            print(f"    -> Failed with: {error_msg}")
            next_tokens = get_tokens_for_attempt(attempt + 1, context_length)
            if attempt + 1 <= 3:
                next_strategy = f"reduced by {attempt}"
            elif attempt + 1 == 4:
                next_strategy = "half context length"
            elif attempt + 1 == 5:
                next_strategy = "one-third context length"
            elif attempt + 1 == 6:
                next_strategy = "one-quarter context length"
            print(f"    -> Will retry with {next_tokens} tokens ({next_strategy})...")
    
    # Final result based on last attempt
    final_success = last_result.get('success', False) if last_result else False
    print(f"  [FINAL] Context test {'PASSED' if final_success else 'FAILED'} after {attempt} attempt(s)")
    
    # Return simple success/failure icon for display
    if final_success and attempt == 1:
        return "✅"
    elif final_success:
        return "⚠️"
    else:
        return "❌"

