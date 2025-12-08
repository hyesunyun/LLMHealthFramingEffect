import argparse
import os

from models.gpt5 import GPT5
from models.claude import Claude
from models.llama3 import Llama3
from models.deepseek import DeepSeek
from models.qwen3 import Qwen3
from models.qwen3_thinking import Qwen3Thinking

from utils import load_json_file, load_jsonl_file, save_dataset_to_json, render_prompt, format_review_abstract, extract_yes_or_no
from tqdm import tqdm
import time
import torch, gc

REQ_TIME_GAP = 5 # seconds to wait between requests to avoid rate limiting
DEFAULT_MAX_NEW_TOKENS = 20000 # arbitrary number for default max tokens
MODELS_WITH_RATE_LIMIT = ["claude_4.5_sonnet"]
REASONING_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", "qwen3_thinking-4B", "qwen3_thinking-30B"]
MODEL_CLASS_MAPPING = {
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

class Filter:
    def __init__(self, model_name: str, input_path: str, output_path: str, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
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
        model_class = MODEL_CLASS_MAPPING[self.model_name]
        if "-" in self.model_name:
            type = model_name.split("-")[-1]
            self.model = model_class(model_type=type)
        else:
            self.model = model_class()

    def __save_data(self, data: list, output_path: str) -> None:
        """
        This method saves the dataset to the output path.

        :params dataset: list of dictionaries to save
        :params output_path: path and file name to save the data

        :return: None
        """
        print(f"Saving the dataset to {output_path}...")
        # save results to a json file
        if output_path.endswith(".jsonl"):
            save_dataset_to_json(data, output_path, jsonl=True)
        elif output_path.endswith(".json"):
            save_dataset_to_json(data, output_path)
    
    def filter_data(self) -> None:
        """
        This method filter the dataset based on model predictions.

        :return None
        """
        # run the task using specified model
        results = []
        print(f"Number of reviews before filtering: {len(self.dataset)}")
        pbar = tqdm(self.dataset, desc="Running filtering process on the dataset")
        for _, example in enumerate(pbar):
            review_abstract_sections = example["ReviewAbstract"]
            formatted_abstract = format_review_abstract(review_abstract_sections)

            is_generic_topic_type_prompt = render_prompt("filter_reviews_by_topic_type", template_dir="./prompts", review_abstract=formatted_abstract)
            if self.is_reasoning_model:
                response, thinking_context = self.model.generate_output(is_generic_topic_type_prompt, max_new_tokens=self.max_new_tokens)
                print(f"[{example["ReviewID"]}] Thinking Context: {thinking_context}")
            else:
                response = self.model.generate_output(is_generic_topic_type_prompt, max_new_tokens=self.max_new_tokens)
            print(f"[{example["ReviewID"]}] Model Response: {response}")

            example["LLMThinkingContext"] = thinking_context if self.is_reasoning_model else ""
            example["LLMRawResponse"] = response
            
            # some cleaning may be needed - extract yes or no from string
            response = extract_yes_or_no(response)

            example["IsSimpleGenericTopic"] = response

            if self.model_name in MODELS_WITH_RATE_LIMIT:
                # add some default time gap to avoid rate limiting
                time.sleep(REQ_TIME_GAP)

            results.append(example)
        # end of loop through the dataset

        # save the intermediary data file
        self.__save_data(results, f"./outputs/filtering_data/{self.input_path.split(".")[0]}_with_topic_type.json")

        # filter out ones that are None or no for IsSimpleGenericTopic
        filtered_data = []
        for example in results:
            is_generic = example["IsSimpleGenericTopic"]
            if is_generic == "yes":
                filtered_data.append(example)
        print(f"Number of reviews after filtering: {len(filtered_data)}")

        # save the filtered data
        self.__save_data(filtered_data, self.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Filtering of Reviews by Topic Type (Simple/General vs. Complex/Specific)")

    parser.add_argument("--model", default="llama3.3_instruct_70B", 
                        choices=["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", 
                                 "llama3.3_instruct_70B", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", 
                                 "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B"], 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--input_path", default="../outputs", help="path to the input file containing review abstracts."),
    parser.add_argument("--output_path", default="./outputs", help="path with file name to save filtered data.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 10 sampled data.")
    
    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for Filter:")
    print(f"Model:    {model_name}")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print(f"Is Debug: {is_debug}")
    print()
    
    filter = Filter(model_name, input_path, output_path, is_debug)
    filter.filter_data()
    gc.collect()
    torch.cuda.empty_cache()
