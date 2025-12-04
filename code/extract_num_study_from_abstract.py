import argparse
import os

from models.gpt5 import GPT5
from models.claude import Claude
from models.llama3 import Llama3
from models.deepseek import DeepSeek
from models.qwen3 import Qwen3
from models.qwen3_thinking import Qwen3Thinking

from utils import load_json_file, load_jsonl_file, save_dataset_to_json, render_prompt, format_review_abstract
from tqdm import tqdm
import time
import torch, gc
import matplotlib.pyplot as plt
import json


DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

REQ_TIME_GAP = 5 # seconds to wait between requests to avoid rate limiting
DEFAULT_MAX_NEW_TOKENS = 5000 # arbitrary number for default max tokens
MODELS_WITH_RATE_LIMIT = ["claude_4.5_sonnet"]
REASONING_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", "qwen3_thinking-4B", "qwen3_thinking-30B"]

class Extractor:
    
    def __init__(self, model_name: str, input_path: str, output_path: str, plot_path: str, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.plot_path = plot_path
        self.is_debug = is_debug

        self.dataset = None
        self.model = None
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS

        self.__load_dataset()
        self.__load_model()
        self.is_reasoning_model = self.model_name in REASONING_MODELS

    def __load_dataset(self) -> None:
        """
        This method loads the dataset (test split)

        :return dataset as a list of dictionaries
        """
        print("Loading the dataset...")
        if self.input_path.endswith(".jsonl"):
            dataset = load_jsonl_file(self.input_path)
        elif self.input_path.endswith(".json"):
            dataset = load_json_file(self.input_path)

        if self.is_debug:
            dataset = dataset[:5] # use only first 5 examples for debugging

        self.dataset = dataset

    def __load_model(self) -> None:
        """
        This method loads the model requested for the task based on the model size.

        :return Model object
        """
        print("Loading the model...")
        model_class_mapping = {
                "gpt-5.1": GPT5,
                "gpt5-mini": GPT5,
                "gpt5-nano": GPT5,
                "claude_4.5_sonnet": Claude,
                "deepseek_distill-qwen32B": DeepSeek,
                "deepseek_distill-llama70B": DeepSeek,
                "llama3.3_instruct_70B": Llama3,
                "qwen3-4B": Qwen3,
                "qwen3-30B": Qwen3,
                "qwen3_thinking-4B": Qwen3Thinking,
                "qwen3_thinking-30B": Qwen3Thinking,
            }
        model_class = model_class_mapping[self.model_name]
        if "-" in self.model_name:
            type = model_name.split("-")[-1]
            self.model = model_class(model_type=type)
        else:
            self.model = model_class()

    def __calculate_accuracy(self, preds, labels) -> dict[str, float]:
        '''
        Calculate accuracy of predictions against labels.
        '''
        assert len(preds) == len(labels)
        total = len(preds)
        correct = sum([1 for p, l in zip(preds, labels) if p == l])
        accuracy = correct / total if total > 0 else 0 

        return {"accuracy": accuracy, "total": total, "correct": correct}

    def __calculate_mean_absolute_error(self, preds, labels) -> dict[str, float]:
        '''
        Calculate mean absolute error of predictions against labels.
        '''
        assert len(preds) == len(labels)
        total = len(preds)
        absolute_errors = [abs(p - l) for p, l in zip(preds, labels)]
        mae = sum(absolute_errors) / total if total > 0 else 0 

        return {"mae": mae, "total": total}

    def __plot_abs_diff_histogram(self, preds, labels) -> dict[str, float]:
        '''
        Plot histogram of absolute differences between predictions and labels.
        '''
        assert len(preds) == len(labels)
        absolute_errors = [abs(p - l) for p, l in zip(preds, labels)]

        plt.hist(absolute_errors, bins=20, edgecolor='black')
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of Absolute Differences')
        # Save the plot as a PDF
        plt.savefig(self.plot_path)
    
    def extract(self) -> None:
        """
        This method extracts number of studies from the review abstract using the loaded model.

        :return None
        """
        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running model extract number of studies on the dataset")
        for _, example in enumerate(pbar):
            review_abstract_sections = example["ReviewAbstract"]
            formatted_abstract = format_review_abstract(review_abstract_sections)

            extracted_num_studies_prompt = render_prompt("extract_num_studies", template_dir="./prompts", review_abstract=formatted_abstract)
            if self.is_reasoning_model:
                response, thinking_context = self.model.generate_output(extracted_num_studies_prompt, max_new_tokens=self.max_new_tokens)
                # print(f"Thinking Context: {thinking_context}")
            else:
                response = self.model.generate_output(extracted_num_studies_prompt, max_new_tokens=self.max_new_tokens)
            # print(f"Model Response: {response}")

            example["LLMThinkingContext"] = thinking_context if self.is_reasoning_model else ""
            example["LLMRawResponse"] = response
            
            # some cleaning may be needed
            response = response.replace("```", "").replace("json", "")
            # convert response from json to dict
            response_dict = json.loads(response)

            # in case of error, skip to next example
            if "error" in response_dict:
                print(f"[ERROR] Model returned an error: {response_dict['error']}")
                example["LLMExtractedNumStudies"] = None
                results.append(example)
                continue
            
            if "NumStudies" not in response_dict:
                num_studies_output = None
            else:
                num_studies_output = response_dict["NumStudies"]
            example["LLMExtractedNumStudies"] = num_studies_output

            if self.model_name in MODELS_WITH_RATE_LIMIT:
                # add some default time gap to avoid rate limiting
                time.sleep(REQ_TIME_GAP)

            results.append(example)
        # end of loop through the dataset

        # extract the predictions and labels
        preds = [ex["LLMExtractedNumStudies"] for ex in results]
        labels = [ex["TrueNumStudies"] for ex in results]

        # calculate and print accuracy and mean absolute error
        accuracy_results = self.__calculate_accuracy(preds, labels)
        mae_results = self.__calculate_mean_absolute_error(preds, labels)
        print(f"Accuracy: {accuracy_results['accuracy']:.4f} ({accuracy_results['correct']}/{accuracy_results['total']})")
        print(f"Mean Absolute Error: {mae_results['mae']:.4f}")

        # histogram of absolute differences
        if self.plot_path is not None:
          self.__plot_abs_diff_histogram(preds, labels)

        # save results to a json file
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True)
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(results, self.output_path)
        # print the number of rows in the results
        print(f"number of rows in the results: {len(results)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Generation of Questions from Cochrane Reviews Using LLMs")

    parser.add_argument("--model", default="llama3.3_instruct_70B", 
                        choices=["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", 
                                 "llama3.3_instruct_70B", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", 
                                 "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B"], 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--input_path", default="../outputs", help="path to the input file containing the effectiveness questions."),
    parser.add_argument("--output_path", default="./outputs", help="path with file name to save outputs.")
    parser.add_argument("--plot_path", default=None, help="(Optional) path with file name to save the histogram plot.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 10 sampled data.")
    
    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    output_path = args.output_path
    plot_path = args.plot_path
    is_debug = args.debug

    print("Arguments Provided for Extractor:")
    print(f"Model:    {model_name}")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print(f"Plot Path: {plot_path}")
    print(f"Is Debug: {is_debug}")
    print()
    
    generator = Extractor(model_name, input_path, output_path, plot_path, is_debug)
    generator.extract()
    gc.collect()
    torch.cuda.empty_cache()
