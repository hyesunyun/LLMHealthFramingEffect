import argparse
import os
from tqdm import tqdm
import time
import json

from utils import load_jsonl_file, load_json_file, save_dataset_to_json, render_prompt, format_review_abstract, \
    extract_json_string, format_messages
from constants import REQ_TIME_GAP, MODELS_WITH_RATE_LIMIT, REASONING_MODELS, MODEL_CLASS_MAPPING, MODELS

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_MAX_NEW_TOKENS = 5000
PROMPT_TEMPLATE_NAME = "generate_simplified_terms"


class Simplifier:

    def __init__(self, model_name: str, input_path: str, output_path: str, max_new_tokens: int,
                 is_debug: bool = False) -> None:
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.is_debug = is_debug

        self.is_reasoning_model = model_name in REASONING_MODELS

        self.dataset = None
        self.model = None
        self.max_new_tokens = max_new_tokens

        self.__load_dataset()
        self.__load_model()

    def __load_dataset(self) -> None:
        print("Loading the dataset...")
        if self.input_path.endswith(".jsonl"):
            dataset = load_jsonl_file(self.input_path)
        elif self.input_path.endswith(".json"):
            dataset = load_json_file(self.input_path)
        else:
            raise ValueError("Input file must be .json or .jsonl")

        if self.is_debug:
            dataset = dataset[:10]

        self.dataset = dataset

    def __load_model(self) -> None:
        print("Loading the model...")
        model_class = MODEL_CLASS_MAPPING[self.model_name]
        if "-" in self.model_name:
            m_type = self.model_name.split("-")[-1]
            self.model = model_class(model_type=m_type)
        else:
            self.model = model_class()

    def simplify_terms(self) -> None:
        results = []
        pbar = tqdm(self.dataset, desc="Running simplification on the dataset")
        for _, example in enumerate(pbar):
            review_title = example["ReviewTitle"]
            review_abstract_sections = example["ReviewAbstract"]
            formatted_abstract = format_review_abstract(review_abstract_sections)

            extracted_terms = example["ExtractedText"]
            intervention = extracted_terms["intervention"]
            condition = extracted_terms["condition"]

            # skip model inference if the extracted values are null/None
            if intervention is None or condition is None:
                example["SimplifiedExtractedText"] = {"simplified_intervention": None, "simplified_condition": None}
            else:
                input_text = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts",
                                        review_title=review_title,
                                        review_abstract=formatted_abstract,
                                        intervention=intervention,
                                        condition=condition)

                messages = format_messages(self.model_name, input_text)

                if self.is_reasoning_model:
                    response, thinking_context = self.model.generate_output(messages, max_new_tokens=self.max_new_tokens)
                else:
                    response = self.model.generate_output(messages, max_new_tokens=self.max_new_tokens)
                    thinking_context = ""

                example["LLMThinkingContext"] = thinking_context
                example["LLMRawResponse"] = response

                try:
                    example["SimplifiedExtractedText"] = json.loads(extract_json_string(response))
                except Exception:
                    example["SimplifiedExtractedText"] = {"simplified_intervention": None, "simplified_condition": None}

            if self.model_name in MODELS_WITH_RATE_LIMIT:
                time.sleep(REQ_TIME_GAP)

            results.append(example)

        print(f"Saving outputs from model - {self.model_name}")
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True)
        else:
            save_dataset_to_json(results, self.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running Simplification of Intervention and Condition Terms from Cochrane Reviews Using LLMs")

    parser.add_argument("--model", default="qwen3-4B",
                        choices=MODELS,
                        help="what model to run",
                        required=True)
    parser.add_argument("--input_path", required=True,
                        help="path to the input json file.")
    parser.add_argument("--output_path", default="./outputs/simplified_results.json",
                        help="path where results should be saved.")
    parser.add_argument("--max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int,
                        help="maximum number of tokens to generate")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction,
                        help="run first 10 rows of data.")

    args = parser.parse_args()

    directory_path = os.path.dirname(args.output_path)
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)

    simplifier = Simplifier(args.model, args.input_path, args.output_path, args.max_new_tokens, args.debug)
    simplifier.simplify_terms()
