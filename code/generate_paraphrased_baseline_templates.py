import argparse
import json
import os

from tqdm import tqdm
import time
import torch, gc

from utils import load_jsonl_file, load_json_file, render_prompt
from constants import REQ_TIME_GAP, MODELS_WITH_RATE_LIMIT, REASONING_MODELS, MODEL_CLASS_MAPPING, MODELS

import tiktoken
from rapidfuzz.distance import Levenshtein

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROMPT_TEMPLATE_NAME = "generate_paraphrased_baseline_question"

class Paraphraser:

    def __init__(self, model_name: str, input_path: str, output_path: str, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.is_debug = is_debug
        self.is_reasoning_model = model_name in REASONING_MODELS
        self.max_new_tokens = 1000 if self.is_reasoning_model else 100

        self.question_templates = self.__load_templates()
        self.model = self.__load_model()

    # Setup
    def __load_templates(self) -> list[dict]:
        print("Loading the dataset...")
        if self.input_path.endswith(".jsonl"):
            dataset = load_jsonl_file(self.input_path)
        elif self.input_path.endswith(".json"):
            dataset = load_json_file(self.input_path)

        if self.is_debug:
            dataset = dataset[:3]

        return dataset

    def __load_model(self):
        print("Loading the model...")
        model_class = MODEL_CLASS_MAPPING[self.model_name]
        if "-" in self.model_name:
            model_type = self.model_name.split("-")[-1]
            return model_class(model_type=model_type)
        return model_class()

    # Helpers
    def _get_gpt_token_distance(question1: str, question2: str, model="gpt-5.1"):
        # Get the correct encoding for the model
        # Note: gpt-5.1 uses the o200k_base encoding
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback if the specific alias isn't in your library version yet
            encoding = tiktoken.get_encoding("o200k_base")

        # Tokenize the sentences or questions
        tokens1 = encoding.encode(question1)
        tokens2 = encoding.encode(question2)

        # Calculate Levenshtein Distance
        distance = Levenshtein.distance(tokens1, tokens2)
        
        return distance, tokens1, tokens2

    # Main entry point
    def paraphrase_templates(self) -> None:
        for template_name in enumerate(tqdm(self.question_templates.keys(), desc="Running paraphrasing")):
            pos_template = self.question_templates[template_name]["positive_question_template"]
            neg_template = self.question_templates[template_name]["negative_question_template"]
            distance, tokens_pos, tokens_neg = self._get_gpt_token_distance(pos_template, neg_template)
            print(f"Distance between:\n'{pos_template}'\nand\n'{neg_template}'\nis: {distance}\n")
            percentage_distance = distance / len(tokens_pos) * 100
            self.question_templates[template_name]["token_level_levenshtein_distance"] = distance
            self.question_templates[template_name]["positive_question_template_token_count"] = len(tokens_pos)
            self.question_templates[template_name]["negative_question_template_token_count"] = len(tokens_neg)
            self.question_templates[template_name]["token_level_levenshtein_distance_percentage"] = percentage_distance

            input_text = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts",
                                   question=pos_template, target_pct=percentage_distance, orig_token_count=len(tokens_pos), target_edits_count=distance)
            
            output = self.model.generate_output(input_text, max_new_tokens=self.max_new_tokens)
            if isinstance(output, tuple):
                output = output[0]
            self.question_templates[template_name]["positive_question_template_paraphrased"] = output

            para_distance, tokens_pos_orig, tokens_pos_paraphrased = self._get_gpt_token_distance(pos_template, output)
            percentage_distance_para = para_distance / len(tokens_pos_orig) * 100
            self.question_templates[template_name]["paraphrased_question_template_token_count"] = len(tokens_pos_paraphrased)
            self.question_templates[template_name]["token_level_levenshtein_distance_paraphrased"] = para_distance
            self.question_templates[template_name]["token_level_levenshtein_distance_percentage_paraphrased"] = percentage_distance_para

            if self.model_name in MODELS_WITH_RATE_LIMIT:
                    time.sleep(REQ_TIME_GAP)

        # save output as json
        with open(self.output_path, "w") as f:
            json.dump(self.question_templates, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running LLM to paraphrase positive question template for baseline evaluation.")

    parser.add_argument("--model", default="qwen3-thinking-4B",
                        choices=MODELS,
                        help="what model to run",
                        required=True)
    parser.add_argument("--input_path", default=DATA_FOLDER_PATH, help="path to the input file with question templates.")
    parser.add_argument("--output_path", default="./outputs", help="path/file name of where the outputs/results should be saved.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run 3 samples.")

    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    output_path = args.output_path
    is_debug = args.debug

    print()
    print("Arguments Provided for Template Paraphraser:")
    print(f"Model:             {model_name}")
    print(f"Input Path:        {input_path}")
    print(f"Output Path:       {output_path}")
    print(f"Is Debug:          {is_debug}")
    print()

    # check if input and output paths exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist. Please provide a valid path to the dataset file.")
    # Get the directory name
    directory_path = os.path.dirname(output_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Output directory path did not exist. Directory was created.")

    paraphraser = Paraphraser(model_name, input_path, output_path, is_debug=is_debug)
    paraphraser.paraphrase_templates()
    gc.collect()
    torch.cuda.empty_cache()
