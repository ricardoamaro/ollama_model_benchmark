TABLE_TEST_PROMPT = (
    "Create a markdown table comparing 'Hello, World!' code in C, Python, Go, PHP, Ruby, Java, JavaScript, Perl, Lisp, Haskell, and Rust. "
    "Don't forget table separator line (e.g., |---|---|) and always to open and close each line of the table with |"
    "Markdown table header needs to be: | Language | Code |"
    "Each row should be a language in a single line, with a column for the language name and another for a minimal, working code block. "
    "Provide only the table. There should be text strictly and only inside the table."
    "Do not use markdown code fences in your response."
)

PYTHON_CODING_PROMPT = (
    "Write a Python function called `sum_even_numbers` that takes a list of integers as input "
    "and returns the sum of all the even numbers in the list. "
    "Provide only the Python function code block itself in the least lines possible. "
    "Do not include any example usage, comments, explanations, conversational filler, or introductory phrases outside the function block. "
    "Do not use markdown code fences in your response. Just show the python code with no other text."
)

SELF_IDENTIFY_PROMPT = (
    "What is your model name or identifier? Respond concisely with only the name or identifier itself. "
    "Do not include any additional sentences, explanations, conversational filler, or introductory phrases."
)

RELEASE_DATE_PROMPT = (
    "What is your model's official release date? Respond concisely with only the date in YYYY-MM-DD format if known, or just the year if not. "
    "Do not include any additional sentences, explanations, conversational filler, or introductory phrases."
)

DEVELOPER_PROMPT = (
    "What is the name of the organization that created you? Respond concisely with only the organization name in less than 3 words."
    "Do not include any additional sentences, explanations, conversational filler, or introductory phrases."
)
