import argparse
import os

from tqdm import tqdm
import time
import json
import random

from utils import load_json_file, load_jsonl_file, save_dataset_to_json
from constants import SEED

class Templater:
    
    def __init__(self, input_path: str, output_path: str, is_debug: bool = False) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.is_debug = is_debug

        self.dataset = None
        self.template = None
        self.__load_dataset()
        self.__load_template()

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

        # only load random 10 samples if in debug mode
        if self.is_debug:
            random.seed(SEED)
            dataset = random.sample(dataset, 10)

        self.dataset = dataset

    def __load_template(self) -> str:
        """
        This method loads the templates from a json file.

        :return template as a list of dict
        """
        # TODO: can add template file path as a parameter
        template_file_path = os.path.join(os.path.dirname(__file__), "./prompts/question_templates.json")
        # for each question type, there is positive_question_template and negative_question_template
        self.template = load_json_file(template_file_path)

    def apply_template(self) -> None:
        """
        This method uses the extracted intervention and condition to generate questions using the templates

        :return None
        """
        results = []
        pbar = tqdm(self.dataset, desc="Running applying template to each sample in the dataset")
        for _, example in enumerate(pbar):
            if "ExtractedText" not in example:
                continue
            
            extracted_text = example["ExtractedText"]
            if "intervention" in extracted_text and "condition" in extracted_text:
                intervention = extracted_text["intervention"]
                condition = extracted_text["condition"]
            else:
                continue

            # loop through each key value pair in the template
            for question_type, templates in self.template.items():
                positive_template = templates["positive_question_template"]
                negative_template = templates["negative_question_template"]

                positive_question = positive_template.format(intervention=intervention, condition=condition)
                negative_question = negative_template.format(intervention=intervention, condition=condition)

                if "Questions" not in example:
                    example["Questions"] = {}
                example["Questions"][question_type] = {
                    "positive_question": positive_question,
                    "negative_question": negative_question
                }
            
            results.append(example)
        # end of loop through the dataset

        # saving outputs to file
        print(f"Saving outputs")
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(data, self.output_path, jsonl=True)
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(data, self.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Apply Template for Question Creation")

    # TODO: potentially can make template file a parameter
    parser.add_argument("--input_path", default="./outputs", help="path to the input json file containing intervention and condition for each Cochrane Review.")
    parser.add_argument("--output_path", default="./outputs", help="directory of where the outputs/results should be saved.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 10 randomly sampled data.")
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for Templater:")
    print(f"Input Path:        {input_path}")
    print(f"Output Path:       {output_path}")
    print(f"Is Debug:          {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    templater = Templater(input_path, output_path, is_debug)
    templater.apply_template()
