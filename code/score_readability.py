import argparse
import os
from tqdm import tqdm
import torch, gc
import random

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from utils import load_jsonl_file, load_json_file, save_dataset_to_json
from constants import SEED

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class ReadabilityScorer:
    """
    Based on https://github.com/chaojiang06/medreadme/tree/main
    This class loads the model and tokenizer to score the readability of a given text.
    The model predicts the CEFR score of the input text.
    """
    
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.__load_model_tokenizer()

    def __load_model_tokenizer(self) -> None:
        """
        This method loads the model and tokenizer to use

        :return Model object
        """
        model_name = "chaojiang06/medreadme_medical_sentence_readability_prediction_CWI"
        # Define a configuration for regression (num_labels=1)
        config = AutoConfig.from_pretrained(model_name, num_labels=1)

        # Load the model with the regression configuration
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def score(self, text: str) -> float:
        """
        This method scores the text using the model that predicts the CEFR score

        :return score as a float
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # The output logits are in outputs.logits
        score = outputs.logits. squeeze().item()  # Get the score as a float
        return score


# Methods outside class for running the scoring for a given input dataset and saving the outputs
def load_dataset(input_path: str, is_debug: bool = False) -> None:
    """
    This method loads the dataset

    :param input_path: path to the input dataset
    :param is_debug: flag to indicate if debug mode is on

    :return dataset as a list of dictionaries
    """
    print("Loading the dataset...")
    if input_path.endswith(".jsonl"):
        dataset = load_jsonl_file(input_path)
    elif input_path.endswith(".json"):
        dataset = load_json_file(input_path)

    # only load random 10 samples if in debug mode
    if is_debug:
        random.seed(SEED)
        dataset = random.sample(dataset, 10)

    return dataset

def score_questions(dataset: list[dict], scorer: ReadabilityScorer, output_path: str) -> None:
    """
    This method scores the questions for their readability

    :param dataset: list of dictionaries containing the dataset
    :param scorer: ReadabilityScorer object
    :param output_path: path to the output file

    :return None
    """
    results = []
    pbar = tqdm(dataset, desc="Running generation on the dataset")
    for _, item in enumerate(pbar):
        questions = item["Questions"]
        item["MedReadMeScores"] = {}
        for key, value in questions.items():
            positive_question = value["positive"]
            negative_question = value["negative"]
            positive_question_score = scorer.score(positive_question)
            negative_question_score = scorer.score(negative_question)
            
            item["MedReadMeScores"][key] = {
                "positive": positive_question_score,
                "negative": negative_question_score
            }
        results.append(item)
    # end of loop through the dataset

    # saving outputs to file
    print(f"Saving outputs")
    if output_path.endswith(".jsonl"):
        save_dataset_to_json(results, output_path, jsonl=True)
    elif output_path.endswith(".json"):
        save_dataset_to_json(results, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Scoring of Readability of Questions")

    parser.add_argument("--input_path", default="./outputs", help="directory and name of file of where the model genrated questions are saved.", required=True)
    parser.add_argument("--output_path", default="./outputs", help="directory of where the outputs/results should be saved.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 10 randomly sampled data.")
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for Reliability Scorer:")
    print(f"Input Path:  {input_path}")
    print(f"Output Path: {output_path}")
    print(f"Is Debug:    {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    scorer = ReadabilityScorer()
    dataset = load_dataset(input_path, is_debug)

    score_questions(dataset, scorer, output_path)
    gc.collect()
    torch.cuda.empty_cache()
