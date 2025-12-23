import argparse
import os

from tqdm import tqdm
import time
import json

from utils import load_jsonl_file, load_json_file, save_dataset_to_json, render_prompt, format_review_abstract, extract_json_string, format_messages
from constants import REQ_TIME_GAP, MODELS_WITH_RATE_LIMIT, REASONING_MODELS, MODEL_CLASS_MAPPING, MODELS

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_MAX_NEW_TOKENS = 5000 # arbitrary number for default max tokens
PROMPT_TEMPLATE_NAME = "intervention_condition_extraction"

class Extractor:
    
    def __init__(self, model_name: str, input_path: str, output_path: str, max_new_tokens: int, is_debug: bool = False) -> None:
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
        """
        This method loads the dataset

        :return dataset as a list of dictionaries
        """
        print("Loading the dataset...")
        if self.input_path.endswith(".jsonl"):
            dataset = load_jsonl_file(self.input_path)
        elif self.input_path.endswith(".json"):
            dataset = load_json_file(self.input_path)

        if self.is_debug:
            dataset = dataset[:10] # use only first 10 examples for debugging

        self.dataset = dataset

    def __load_model(self) -> None:
        """
        This method loads the model requested for the task based on the model size.

        :return Model object
        """
        print("Loading the model...")
        model_class = MODEL_CLASS_MAPPING[self.model_name]
        if "-" in self.model_name:
            type = model_name.split("-")[-1]
            self.model = model_class(model_type=type)
        else:
            self.model = model_class()

    def extract_intervention_condition(self) -> None:
        """
        This method extracts the intervention and coniditon of the review in plain language for question generation.

        :return None
        """
        results = []
        pbar = tqdm(self.dataset, desc="Running extraction on the dataset")
        for _, example in enumerate(pbar):
            review_abstract_sections = example["ReviewAbstract"]
            formatted_abstract = format_review_abstract(review_abstract_sections)
            input = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts",
                                  review_abstract=formatted_abstract)

            # format messages
            messages = format_messages(self.model_name, input)

            if self.is_reasoning_model:
                response, thinking_context = self.model.generate_output(messages, max_new_tokens=self.max_new_tokens)
                # print(f"[{example['ReviewID']}] Thinking Context: {thinking_context}")
            else:
                response = self.model.generate_output(messages, max_new_tokens=self.max_new_tokens)
            # print(f"[{example['ReviewID']}] Model Response: {response}")

            example["LLMThinkingContext"] = thinking_context if self.is_reasoning_model else ""
            example["LLMRawResponse"] = response
            # ExtractedText are in JSON format
            try:
                example["ExtractedText"] = json.loads(extract_json_string(response))
            except:
                example["ExtractedText"] = {"intervention": None, "condition": None}

            if self.model_name in MODELS_WITH_RATE_LIMIT:
                # add some default time gap to avoid rate limiting
                time.sleep(REQ_TIME_GAP)

            results.append(example)
        # end of loop through the dataset

        # saving outputs to file
        print(f"Saving outputs from model - {self.model_name}")
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True)
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(results, self.output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Extraction of Interventions and Conditions from Cochrane Reviews Using LLMs")

    parser.add_argument("--model", default="llama3.3_instruct_70B", 
                        choices=MODELS, 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--input_path", default="./outputs", help="path to the input json file containing Cochrane Reviews.")
    parser.add_argument("--output_path", default="./outputs", help="directory of where the outputs/results should be saved.")
    parser.add_argument("--max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int, help="maximum number of tokens to generate for the key question")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run first 10 rows of data.")
    
    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    output_path = args.output_path
    max_new_tokens = args.max_new_tokens
    is_debug = args.debug

    print("Arguments Provided for Extractor:")
    print(f"Model:             {model_name}")
    print(f"Input Path:        {input_path}")
    print(f"Output Path:       {output_path}")
    print(f"Max Output Tokens: {max_new_tokens}")
    print(f"Is Debug:          {is_debug}")
    print()

    # Get the directory name
    directory_path = os.path.dirname(output_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Output directory path did not exist. Directory was created.")
    
    extractor = Extractor(model_name, input_path, output_path, max_new_tokens, is_debug)
    extractor.extract_intervention_condition()
