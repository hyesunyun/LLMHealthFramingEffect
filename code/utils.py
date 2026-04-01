
import csv
import json
import sys
from jinja2 import Environment, FileSystemLoader
import os
import re
from typing import Optional

def load_csv_file(file_path: str) -> list[dict]:
    """
    This method loads a CSV file and returns the data as a list of dictionaries

    :param file_path: path to the CSV file

    :return data as a list of dictionaries
    """
    csv.field_size_limit(sys.maxsize) # this is needed for long text in the csv file, such as the review abstract
    with open(file_path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def load_json_file(file_path: str) -> list[dict]:
    """
    This method loads a JSON file and returns the data as a list of dictionaries

    :param file_path: path to the JSON file

    :return data as a list of dictionaries
    """
    with open(file_path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return data

def load_jsonl_file(file_path: str) -> list[dict]:
    """
    This method loads a JSONL file and returns the data as a list of dictionaries

    :param file_path: path to the JSONL file

    :return data as a list of dictionaries
    """
    data = []
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_dataset_to_json(dataset: list[dict], file_path: str, jsonl: bool = False, columns_to_drop: list[str] | None = None) -> None:
    """
    This method saves a dataset (dictionary) in json file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save
    :param jsonl: whether to save as jsonl file
    :param columns_to_drop: list of columns to drop from the dataset before saving
    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
    with open(file_path, "w", encoding='utf-8') as file:
        if jsonl:
            for entry in dataset:
                file.write(json.dumps(entry) + "\n")
        else:
            json.dump(dataset, file, indent=4)


def save_dataset_to_csv(dataset: list[dict], file_path: str, columns_to_drop: list[str] | None = None) -> None:
    """
    This method saves a dataset (dictionary) in csv file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
    keys = dataset[0].keys()
    with open(file_path, "w", newline='', encoding='utf-8') as file:
        dict_writer = csv.DictWriter(file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)

def render_prompt(template_name: str, template_dir: str = "templates", **kwargs) -> str:
    """
    This method loads a Jinja2 template from the specified directory, then renders it with the provided keyword arguments.
    This is useful for generating dynamic content based on templates.

    :param template_name: Name of the template file to load.
    :param template_dir: Directory where the templates are stored.
    :param kwargs: Keyword arguments to pass to the template for rendering.

    :return: Rendered template as a string.
    """
    try:
        template_dir = os.path.join(os.path.dirname(__file__), template_dir)
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(f"{template_name}.jinja2")
        return template.render(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error rendering prompt '{template_name}': {e}")
    
def format_review_abstract(review_abstract_sections: list) -> str:
        """
        This method formats the review abstract to ensure it is just a string of all headers with corresponding text.

        :param review_abstract: The raw review abstract sections in a list
        :return: Formatted review abstract
        """
        formatted_abstract = ""
        # iterate through the sections and format them
        for section in review_abstract_sections:
            heading = section["heading"]
            text = section["text"]
            formatted_abstract += f"{heading}: {text}\n"
        return formatted_abstract.strip()

def extract_json_string(text: str) -> Optional[str]:
    """
    Extracts a bare JSON-like string content (enclosed in { and }) 
    from a larger text block, assuming the JSON block is the largest 
    contiguous block of curly braces.

    Args:
        text: The input string potentially containing a bare JSON block.

    Returns:
        The extracted JSON-like string, or None if no matching block is found.
    """
    # Regex pattern to find content between the first { and the last }
    # This pattern is greedy (*), matching the largest possible block.
    # The 's' flag (re.DOTALL) allows '.' to match newlines.
    pattern = r"\{.*\}"
    
    match = re.search(pattern, text, re.DOTALL)

    if match:
        # The entire match is the extracted JSON string
        return match.group(0).strip()
    else:
        return None
    
def extract_yes_or_no(text: str) -> str | None:
    """
    Scans a string and returns 'yes' or 'no' if found, ignoring case.

    Args:
        text: The input string to search.

    Returns:
        'yes' or 'no' (lowercase) if found, otherwise None.
    """
    # Use re.search to find the first occurrence of 'yes' or 'no'.
    # The pattern is:
    # 1. (yes|no): A capturing group matching either 'yes' or 'no'.
    # 2. re.IGNORECASE: Makes the search case-insensitive (e.g., 'Yes', 'NO', 'yEs').
    
    match = re.search(r'(yes|no)', text, re.IGNORECASE)

    if match:
        # Return the matched string, converted to lowercase for consistency
        return match.group(1).lower()
    else:
        # Return None if neither 'yes' nor 'no' is found
        return None

def remove_columns(data: list[dict], columns_to_drop: list[str]) -> list[dict]:
    """
    Removes columns or fields within each dict in a list. This is to remove unnecessary columns from a dataset.

    Args:
        data: A list of dict
        columns_to_drop: A list of string names of columns to remove/drop

    Returns:
        a copy of the dataset provided without columns specified
    """
    # using list comprehensionß
    new_data = [
        {k: v for k, v in d.items() if k not in columns_to_drop}
        for d in data
    ]
    return new_data

def format_messages(model_name: str, user_input: str) -> list[dict]:
    """
    Formats user messages for a given model

    Args:
        model_name: string name of the model being used
        user_input: string of the user input

    Returns:
        a formatted messages for model input
    """
    # Define formatting strategies
    formats = {
        "gpt-5": lambda x: [{"role": "user", "content": [{"type": "input_text", "text": x}]}],
        "deepseek": lambda x: [{"role": "user", "content": f"{x}"}, {"role": "assistant", "content": "<think>\n"}],
        "gemini": lambda x: x,
    }

    # Find a match or use a default fallback
    for key, formatter in formats.items():
        if key in model_name:
            return formatter(user_input)

    # Default case
    return [{"role": "user", "content": user_input}]
    