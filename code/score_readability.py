import argparse
import os
from tqdm import tqdm
import torch, gc

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from utils import load_json_file, save_dataset_to_json

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class ReadabilityScorer:
    
    def __init__(self, input_path: str, field_name: str, output_path: str, is_debug: bool = False) -> None:
        self.input_path = input_path
        self.fied_name = field_name
        self.output_path = output_path
        self.is_debug = is_debug

        self.dataset = None
        self.model = None
        self.tokenizer = None
        
        self.__load_dataset()
        self.__load_model_tokenizer()

    def __load_dataset(self) -> None:
        """
        This method loads the dataset

        :return dataset as a list of dictionaries
        """
        print("Loading the dataset...")
        dataset = load_json_file(self.input_path)
        if self.is_debug:
            if len(dataset) > 10:
                dataset = dataset[:10] # use only first 10 examples for debugging
            print(f"Debug mode is ON. Using only {len(dataset)} examples for testing.")

        self.dataset = dataset

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

    def __score(self, text: str) -> float:
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

    def score_questions(self) -> None:
        """
        This method scores the questions for their readability

        :return None
        """
        results = []
        pbar = tqdm(self.dataset, desc="Running generation on the dataset")
        for _, item in enumerate(pbar):
            if self.field_name not in item:
                print(f"Skipping example with ReviewID {item['ReviewID']} as there is no {self.field_name}.")
                continue
            questions = item[self.field_name]
            # check if questions are null
            if questions["positive"] is None and questions["negative"] is None:
                item[f"{self.field_name}_MedReadMeScore"] = {
                    "positive": None,
                    "negative": None
                }
                print(f"Skipping example with ReviewID {item['ReviewID']} as it has no Questions.")
                continue
            positive_question = questions["positive"]
            negative_question = questions["negative"]
            
            positive_question_score = self.__score(positive_question)
            negative_question_score = self.__score(negative_question)
            item[f"{self.field_name}_MedReadMeScore"] = {
                "positive": positive_question_score,
                "negative": negative_question_score
            }
            results.append(item)
        # end of loop through the dataset

        # saving outputs to file
        print("Saving outputs from Readability model")

        # file name
        file_name = self.input_path.split("/")[-1].split(".")[0]
        output_file_name = f"{file_name}_with_readability_scores"

        # convert into json
        json_file_path = f"{self.output_path}/{output_file_name}.json"
        save_dataset_to_json(results, json_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Scoring of Readability")

    parser.add_argument("--input_path", default="./outputs", help="directory and name of file of where the model genrated questions are saved.", required=True)
    parser.add_argument("--field_name", default="./outputs", help="name of field in the json file that contains the questions to be scored.")
    parser.add_argument("--output_path", default="./outputs", help="directory of where the outputs/results should be saved.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 50 randomly sampled data.")
    
    args = parser.parse_args()

    input_path = args.input_path
    field_name = args.field_name
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for Reliability Scorer:")
    print(f"Input Path:  {input_path}")
    print(f"Field Name:  {field_name}")
    print(f"Output Path: {output_path}")
    print(f"Is Debug:    {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    scorer = ReadabilityScorer(input_path, field_name, output_path, is_debug)
    scorer.score_questions()
    gc.collect()
    torch.cuda.empty_cache()
