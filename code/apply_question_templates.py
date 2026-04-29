import argparse
import os

from tqdm import tqdm
import random

from score_readability import ReadabilityScorer

from utils import load_json_file, load_jsonl_file, save_dataset_to_json
from constants import SEED

FULL_QUESTION_TYPES = [
        "effectiveness", "efficacy", "safety", "studies", 
        "timepressure", "cost", "family", "friend", 
        "testimonials", "journals", "ai", "doctor", "multiturn"
    ]

class Templater:
    
    def __init__(self, input_path: str, output_path: str, intervention_condition_key: str, question_types: list = FULL_QUESTION_TYPES, run_scoring: bool = False, is_debug: bool = False) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.intervention_condition_key = intervention_condition_key
        self.question_types = question_types
        self.run_scoring = run_scoring
        self.is_debug = is_debug

        self.dataset = None
        self.template = None
        self.scorer = None
        self.__load_dataset()
        self.__load_template()
        
        if self.run_scoring:
            self.scorer = ReadabilityScorer()

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
        
        # only include rows that are relevant for this task
        dataset = [row for row in dataset if self.intervention_condition_key in row]
        self.dataset = dataset

    def __load_template(self) -> str:
        """
        This method loads the templates from a json file.

        :return template as a list of dict
        """
        # TODO: can add template file path as a parameter
        template_file_path = os.path.join(os.path.dirname(__file__), "./prompts/question_templates.json")
        # for each question type, there is positive_question_template, negative_question_template, and paraphrased_positive_question_template
        templates = load_json_file(template_file_path)
        # filtering to only question types we want
        self.template = {key: templates[key] for key in self.question_types if key in templates}

    def apply_template(self) -> None:
        """
        This method uses the extracted intervention and condition to ReadabilityScorergenerate questions using the templates

        :return None
        """
        results = []
        pbar = tqdm(self.dataset, desc="Running applying template to each sample in the dataset")
        for _, example in enumerate(pbar):
            if self.intervention_condition_key not in example:
                continue

            intervention_condition = example[self.intervention_condition_key]

            if isinstance(intervention_condition, str):
                print(f"[{example['ReviewID']}]: LLM extracted text is not in the correct format.")
                continue
            elif isinstance(intervention_condition, dict) and "intervention" in intervention_condition and "condition" in intervention_condition:
                intervention = intervention_condition["intervention"]
                condition = intervention_condition["condition"]

                if intervention is None or condition is None:
                    print(f"[{example['ReviewID']}]: intervention or condition is None.")
                    continue
            else:
                print(f"[{example['ReviewID']}]: error with LLM extracted text/simplified text.")
                continue

            # loop through each key value pair in the template
            for question_type, templates in self.template.items():
                positive_template = templates["positive_question_template"]
                negative_template = templates["negative_question_template"]
                paraphrased_positive_template = templates["paraphrased_positive_question_template"]

                positive_question = positive_template.format(intervention=intervention, condition=condition)
                negative_question = negative_template.format(intervention=intervention, condition=condition)
                paraphrased_positive_question = paraphrased_positive_template.format(intervention=intervention, condition=condition)

                if "Questions" not in example:
                    example["Questions"] = {}
                example["Questions"][question_type] = {
                    "positive_question": positive_question,
                    "negative_question": negative_question,
                    "paraphrased_positive_question": paraphrased_positive_question
                }

                # add readability (MedReadMe) scoring if specified
                if self.run_scoring:
                    positive_question_score = self.scorer.score(positive_question)
                    negative_question_score = self.scorer.score(negative_question)
                    paraphrased_positive_question_score = self.scorer.score(paraphrased_positive_question)

                    if "MedReadMeScores" not in example:
                        example["MedReadMeScores"] = {}
                    example["MedReadMeScores"][question_type] = {
                        "positive_question": positive_question_score,
                        "negative_question": negative_question_score,
                        "paraphrased_positive_question": paraphrased_positive_question_score
                    }
            
            results.append(example)
        # end of loop through the dataset

        # cleaning up
        columns_to_drop = ["LLMThinkingContext", "LLMRawResponse"]

        # saving outputs to file
        print(f"Saving outputs")
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True, columns_to_drop=columns_to_drop)
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(results, self.output_path, jsonl=False, columns_to_drop=columns_to_drop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Apply Template for Question Creation")

    # TODO: potentially can make template file a parameter
    parser.add_argument("--input_path", default="./outputs", help="path to the input json file containing intervention and condition for each Cochrane Review.")
    parser.add_argument("--output_path", default="./outputs", help="directory of where the outputs/results should be saved.")
    parser.add_argument("--intervention_condition_key", default="ExtractedText", help="the key in the input json file that contains the intervention and condition information to use. Default is 'ExtractedText'.")
    parser.add_argument("--question_types", nargs="+", default=FULL_QUESTION_TYPES, help="the types of questions to create from question_templates.json.")
    # do --no-run_scoring for explicit False
    parser.add_argument("--run_scoring", action=argparse.BooleanOptionalAction, help="whether to run readability scoring after applying templates.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 10 randomly sampled data.")
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    intervention_condition_key = args.intervention_condition_key
    question_types = args.question_types
    run_scoring = args.run_scoring
    is_debug = args.debug

    print("Arguments Provided for Templater:")
    print(f"Input Path:                 {input_path}")
    print(f"Output Path:                {output_path}")
    print(f"Intervention/Condition Key: {intervention_condition_key}")
    print(f"Question Types:             {question_types}")
    print(f"Run Readability Scoring:    {run_scoring}")
    print(f"Is Debug:                   {is_debug}")
    print()

    # Get the directory name
    directory_path = os.path.dirname(output_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Output directory path did not exist. Directory was created.")
    
    templater = Templater(input_path, output_path, intervention_condition_key, question_types, run_scoring, is_debug)
    templater.apply_template()
