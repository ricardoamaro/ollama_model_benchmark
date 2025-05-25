# Ollama Open Source AI Model Benchmark

## What is this project for?

This project provides a benchmarking suite for evaluating Ollama language models on coding-related tasks. It measures model performance on tasks such as coding understanding, table generation and Python coding, reporting metrics like load time, VRAM usage, context window size, and generation speed. The results are output as a markdown report for easy comparison between models.

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ricardoamaro/ollama_model_benchmark.git
   cd ollama_model_benchmark
   ```

2. **Install Python dependencies:**
   Make sure you have Python 3.8+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Ollama:**
   - Download and install Ollama from [https://ollama.com/](https://ollama.com/)
   - Start the Ollama server (default: `http://localhost:11434`)
   - Ensure you have the models you want to benchmark installed in Ollama.

## Running the Benchmarks

Run the main benchmarking script from the command line:

```bash
python ollama_model_benchmark.py
```

### Command-line options

- `--ollama-url URL` &nbsp;&nbsp;&nbsp;&nbsp; Ollama server URL (default: http://localhost:11434)
- `--output FILE` &nbsp;&nbsp;&nbsp;&nbsp; Output markdown file (default: results/ollama_benchmark_results_<timestamp>.md)
- `--models MODEL [MODEL ...]` &nbsp;&nbsp;&nbsp;&nbsp; Specific model names to benchmark (default: all installed)
- `--no-table-task` &nbsp;&nbsp;&nbsp;&nbsp; Skip the table task benchmark
- `--no-coding-task` &nbsp;&nbsp;&nbsp;&nbsp; Skip the Python coding task benchmark
- `--verbose` &nbsp;&nbsp;&nbsp;&nbsp; Enable verbose output
- `--metadata` &nbsp;&nbsp;&nbsp;&nbsp; Show model metadata for debugging

Example:
```bash
python ollama_model_benchmark.py --models llama2 mistral --verbose
```

Benchmark results will be saved in the `results/` directory as a markdown file.

## Prompts used and Results table


```
TABLE_TEST_PROMPT:
Create a markdown table comparing 'Hello, World!' code in C, Python, Go, PHP, Ruby, Java, JavaScript, Perl, Lisp, Haskell, and Rust. Don't forget table separator line (e.g., |---|---|) and always to open and close each line of the table with |Markdown table header needs to be: | Language | Code |Each row should be a language in a single line, with a column for the language name and another for a minimal, working code block. Provide only the table. There should be text strictly and only inside the table.Do not use markdown code fences in your response.

PYTHON_CODING_PROMPT:
Write a Python function called `sum_even_numbers` that takes a list of integers as input and returns the sum of all the even numbers in the list. Provide only the Python function code block itself in the least lines possible. Do not include any example usage, comments, explanations, conversational filler, or introductory phrases outside the function block. Do not use markdown code fences in your response. Just show the python code with no other text.

SELF_IDENTIFY_PROMPT:
What is your model name or identifier? Respond concisely with only the name or identifier itself. Do not include any additional sentences, explanations, conversational filler, or introductory phrases.

RELEASE_DATE_PROMPT:
What is your model's official release date? Respond concisely with only the date in YYYY-MM-DD format if known, or just the year if not. Do not include any additional sentences, explanations, conversational filler, or introductory phrases.

DEVELOPER_PROMPT:
What is the name of the organization that created you? Respond concisely with only the organization name in less than 3 words.Do not include any additional sentences, explanations, conversational filler, or introductory phrases.

```


| Model | Self-ID (resp) | Disk Size | VRAM Usage | License (metadata) | Developer (resp) | Params (K) | Context Len | Release Date | Load (s) | Table Gen (s) | Table Tok/s (eval) | Table FT (s) | Code Gen (s) | Code Tok/s (eval) | Code FT (s) | Context Test | Table Test | Coding Test | 
|-------|:-------:|:---------:|:----------:|:-------:|:----------------:|:----------:|:-----------:|:----------------------:|:------------:|:--------:|:-------------:|:------------------:|:------------:|:------------:|:-----------------:|:-----------:|:-----------:|:-------------:|
| `deepseek-coder-v2:latest` | `My model name or identifier is "GPT-3".` | 8.29 GB | 11.20 GB |  | DeepSeek. | 15706484K | 163840 | 2023 | 6.298 | 5.011 | 62.42 | 0.380 | 1.225 | 65.68 | 0.728 | ✅ | ✅ | ✅ |
| `gemma3:latest` | `Gemini Pro 1.0` | 3.11 GB | 5.61 GB |  | Google DeepMind | 4299915K | 131072 | 2023-03-14 | 0.671 | 5.149 | 42.36 | 0.463 | 1.759 | 43.19 | 0.672 | ✅ | ✅ | ✅ |
| `dolphin3:latest` | `Dolphin` | 4.58 GB | 6.51 GB | llama3.1 | Google AI | 8030277K | 131072 | 2020-05-20 | 4.386 | 7.190 | 29.96 | 0.674 | 1.505 | 31.07 | 0.631 | ✅ | ✅ | ✅ |
| `granite3.3:latest` | `Granite` | 4.60 GB | 7.33 GB | apache-2.0 | IBM | 8170864K | 131072 | 2023-03 | 4.138 | 8.959 | 27.31 | 0.968 | 1.956 | 28.48 | 0.832 | ✅ | ✅ | ✅ |
| `deepcoder:latest` | `DeepSeek-R1` | 8.37 GB | 10.81 GB | mit | OpenAI | 14770033K | 131072 |  | 31.926 | 205.280 | 14.12 | 1.206 | 23.311 | 16.54 | 1.239 | ✅ | ✅ | ✅ |
| `deepseek-r1:32b` | `gpt-3.5-turbo` | 18.49 GB | 21.56 GB |  | OpenAI | 32763876K | 131072 |  | 88.115 | 124.266 | 7.70 | 2.816 | 55.123 | 7.87 | 2.183 | ✅ | ✅ | ✅ |
| `qwen3:latest` | `Qwen` | 4.87 GB | 7.06 GB | apache-2.0 | Alibaba Cloud | 8190735K | 40960 | 2024 | 8.303 | 18.830 | 27.74 | 0.667 | 16.093 | 28.01 | 0.808 | ✅ | ✅ | ✅ |
| `qwq:latest` | `Qwen` | 18.49 GB | 21.56 GB | apache-2.0 |  | 32763876K | 40960 |  | 17.547 | 248.249 | 7.21 | 2.823 | 56.691 | 7.43 | 2.316 | ✅ | ✅ | ✅ |
| `mixtral:latest` | `"Midjourney Language Model"` | 24.63 GB | 23.41 GB | apache-2.0 | ¡Growing AI! | 46702792K | 32768 |  2021-06-17 | 42.507 | 13.439 | 19.68 | 2.511 | 3.399 | 20.52 | 2.006 | ✅ | ✅ | ✅ |
| `codellama:latest` | `My name is LLaMA` | 3.56 GB | 8.77 GB |  | Google | 6738546K | 16384 | 2023-04-15 | 3.496 | 7.255 | 37.55 | 0.540 | 1.597 | 39.28 | 0.717 | ✅ | ✅ | ✅ |
| `codellama:13b` | `LLaMA` | 6.86 GB | 14.53 GB |  | Learning Lodge | 13016028K | 16384 | 2019-06-28 | 5.654 | 11.284 | 20.73 | 1.039 | 2.968 | 21.54 | 1.257 | ✅ | ✅ | ✅ |
| `phi4:latest` | `Phi` | 8.43 GB | 11.51 GB | mit | OpenAI | 14659507K | 16384 | 2023-10-11 | 6.697 | 15.441 | 16.34 | 1.247 | 2.727 | 17.11 | 1.133 | ✅ | ✅ | ✅ |
| `codegemma:7b` | `GPT-4` | 4.67 GB | 9.43 GB |  | Google | 8537680K | 8192 | 2023-02-14 | 4.250 | 8.572 | 30.96 | 0.745 | 1.792 | 32.01 | 0.805 | ✅ | ✅ | ✅ |
| `gemma3:12b` | `Gemini 1.5 Pro` | 7.59 GB | 11.02 GB |  | Google. | 12187079K | 131072 | 2022-09-06 | 5.004 | 14.680 | 17.97 | 1.043 | 3.645 | 18.38 | 0.978 | ✅ | ❌ | ✅ |
| `granite-code:latest` | `My model name is "Tacotron 2".` | 1.86 GB | 5.53 GB | apache-2.0 | OpenAI | 3482503K | 128000 | 2023-05-15 | 7.514 | 4.120 | 58.37 | 0.465 | 2.510 | 60.13 | 0.581 | ❌ | ✅ | ❌ |

Saved results to results/ollama_benchmark_results_2025-05-26_12-20-02.md

**Note:** Interesting from this table are the stochastic responses that are sometimes unpredictable and even misleading in the columns "(resp)". For instance `deepseek-coder-v2:latest` responded as Self-ID: `My model name or identifier is "GPT-3".`, while some struggle sometimes to identify their developer.


## How to Contribute

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes, following PEP8 and best practices.
4. Add or update tests as appropriate.
5. Submit a pull request with a clear description of your changes.

For major changes, please open an issue first to discuss your proposed changes.

## About the Author

Ricardo Amaro

Licence GPLv2 - gnu-gpl-v2.0.md

---

*This project is not affiliated with Ollama. For more information about Ollama, visit [https://ollama.com/](https://ollama.com/).*
