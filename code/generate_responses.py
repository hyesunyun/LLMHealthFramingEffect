import argparse
import os

from tqdm import tqdm
import time
import torch, gc

from utils import load_jsonl_file, load_json_file, save_dataset_to_json, render_prompt, format_review_abstract, extract_json_string
from constants import REQ_TIME_GAP, MODELS_WITH_RATE_LIMIT, REASONING_MODELS, MODEL_CLASS_MAPPING, MODELS

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_MAX_NEW_TOKENS = 500 # arbitrary number for default max tokens
PROMPT_TEMPLATE_NAME = "question_answering"

class Generator:

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

    def __format_rct_inputs(self, rct_inputs: list[dict]) -> list[dict]:
        """
        Function to format RCT inputs for the prompt.
        Only includes title and abstract from each RCT.
        
        :param rct_inputs: list of dictionaries containing RCT inputs

        :return: list of dictionaries containing formatted RCT inputs
        """
        formatted_inputs = []
        for rct in rct_inputs:
            title  = rct["Title"]
            abstract = rct["Abstract"]
            formatted_inputs.append({
                "title": title,
                "abstract": abstract
            })
        return formatted_inputs
    
    def __split_multiturn_question(self, question: str) -> list[str]:
        """
        Split multiturn question into individual questions.
        
        :param question: multiturn question in string format

        :return: list of individual questions
        """
        questions = question.split("\n\n")
        questions = [q.strip() for q in questions if q.strip()]
        return questions

    def __get_answer(self, question: str, rct_inputs: list[dict]) -> tuple[str, str | None]:
        """
        Get answer from the model for the question and RCT inputs.
        
        :param question: question in string format
        :param rct_inputs: list of dictionaries containing RCT inputs

        :return: response from the model and reasoning trace if model is reasoning model, else None
        """
        input = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts", question=question, abstracts=rct_inputs)

        if self.is_reasoning_model:
            response, thinking_context = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)
        else:
            response = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)

        return response, thinking_context if self.is_reasoning_model else None

    def generate_answers(self) -> None:
        """
        This method generates answers to the questions and provided RCT inputs using the specified model.

        :return None
        """
        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running generation on the dataset")
        for _, example in enumerate(pbar):
            rct_inputs = self.__format_rct_inputs(example["Inputs"])
            # For manual questions:
            questions = example["Questions"]
            for i, questions_dict in enumerate(questions):
                positive_question = questions_dict["positive"]
                negative_question = questions_dict["negative"]

                for question in [positive_question, negative_question]:
                    if i != "multiturn":                        
                        response, thinking_context = self.__get_answer(question, rct_inputs)
                        # TODO: remove after debugging
                        print("Thinking Context:")
                        print(thinking_context) 
                        print("Model Response:")
                        print(response)
                        questions[i]["model_answer"] = response.strip()
                    else:
                        multiturn_questions = self.__split_multiturn_question(question)

                        # TODO: need to simulate multi-turn interaction here
                        # input to next turn includes previous Q&As
                        for j, q in enumerate(multiturn_questions):
                            print(f"Multiturn Question {j+1}: {q}")
                            response, thinking_context = self.__get_answer(q, rct_inputs)

                    if self.model_name in MODELS_WITH_RATE_LIMIT:
                        # add some default time gap to avoid rate limiting
                        time.sleep(REQ_TIME_GAP)

            example["ModelGeneratedAnswersWithQuestions"] = questions
            del example["Questions"]  # remove the original questions from the example
            results.append(example)
        # end of loop through the dataset

        # saving outputs to file
        print(f"Saving outputs from model - {self.model_name}")
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True)
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(results, self.output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Generation of Answers to Questions from RCTs Using LLMs")

    parser.add_argument("--model", default="llama3.3_instruct_70B", 
                        choices=MODELS, 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--input_path", default=DATA_FOLDER_PATH, help="path to the input dataset file with RCTs and questions.")
    parser.add_argument("--output_path", default="./outputs", help="path/file name of where the outputs/results should be saved.")
    parser.add_argument("--max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int, help="maximum number of tokens to generate for the answers.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run 3 instances from the dataset.")
    
    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    output_path = args.output_path
    max_new_tokens = args.max_new_tokens
    is_debug = args.debug

    print("Arguments Provided for Answer Generation:")
    print(f"Model:             {model_name}")
    print(f"Input Path:        {input_path}")
    print(f"Output Path:       {output_path}")
    print(f"Max Output Tokens: {max_new_tokens}")
    print(f"Is Debug:          {is_debug}")
    print()

    # check if input and output paths exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist. Please provide a valid path to the dataset file.")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")

    generator = Generator(model_name, input_path, output_path, max_new_tokens, is_debug)
    generator.generate_answers()
    gc.collect()
    torch.cuda.empty_cache()
