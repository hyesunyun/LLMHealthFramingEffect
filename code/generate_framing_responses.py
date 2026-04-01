import argparse
import os

from tqdm import tqdm
import time
import torch, gc

from utils import load_jsonl_file, load_json_file, save_dataset_to_json, render_prompt, format_messages
from constants import REQ_TIME_GAP, MODELS_WITH_RATE_LIMIT, REASONING_MODELS, MODEL_CLASS_MAPPING, MODELS, BATCH_API_MODELS, HF_BATCH_MODELS

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_MAX_NEW_TOKENS = 8000
PROMPT_TEMPLATE_NAME = "question_answering"

class Generator:

    def __init__(self, model_name: str, input_path: str, output_path: str, max_new_tokens: int, batch_size: int = 1, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.is_debug = is_debug
        self.is_reasoning_model = model_name in REASONING_MODELS
        self.max_new_tokens = max_new_tokens

        self.dataset = self.__load_dataset()
        self.model = self.__load_model()

    # Setup
    def __load_dataset(self) -> list[dict]:
        print("Loading the dataset...")
        if self.input_path.endswith(".jsonl"):
            dataset = load_jsonl_file(self.input_path)
        elif self.input_path.endswith(".json"):
            dataset = load_json_file(self.input_path)

        if self.is_debug:
            dataset = dataset[:4]

        return dataset

    def __load_model(self):
        print("Loading the model...")
        model_class = MODEL_CLASS_MAPPING[self.model_name]
        if "-" in self.model_name:
            model_type = self.model_name.split("-")[-1]
            return model_class(model_type=model_type)
        return model_class()

    # Formatting helpers
    def __format_rct_inputs(self, rct_inputs: list[dict]) -> list[dict]:
        """Format RCT inputs (e.g., for chat template) - currently just extracts title and abstract."""
        return [{"title": rct["Title"], "abstract": rct["Abstract"]} for rct in rct_inputs]

    def __build_messages(self, question: str, rct_inputs: list[dict]) -> list[dict]:
        """Render prompt template and format into model-specific messages."""
        input_text = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts",
                                   question=question, abstracts=rct_inputs)
        return format_messages(self.model_name, input_text)

    # Model interaction
    def __call_model(self, messages: list[dict]) -> str:
        """Call model and return only the response text. Handles reasoning models (which return tuples) transparently."""
        output = self.model.generate_output(messages, max_new_tokens=self.max_new_tokens)
        if isinstance(output, tuple):
            return output[0]
        return output

    # Question processing
    def __process_single_turn(self, question: str, rct_inputs: list[dict]) -> str:
        """Get answer for a single non-multiturn question."""
        messages = self.__build_messages(question, rct_inputs)
        return self.__call_model(messages).strip()

    def __process_multiturn(self, question_text: str, rct_inputs: list[dict]) -> tuple[list[str], list[str]]:
        """Run a multiturn conversation. Splits the input question into sub-questions, runs them sequentially, and collects responses."""
        questions = [q.strip() for q in question_text.split("\n\n") if q.strip()]
        responses = []
        messages = None

        for index, q in enumerate(questions):
            if index == 0:
                messages = self.__build_messages(q, rct_inputs)
            else:
                messages.append(format_messages(self.model_name, q)[0])

            response = self.__call_model(messages)
            responses.append(response)

            if "gpt-5" not in self.model_name:
                messages.append({"role": "assistant", "content": response})

        return questions, responses

    def __process_sample_questions(self, rct_inputs: list[dict], questions: dict) -> dict:
        """Process all question types (single-turn + multiturn) for one sample sequentially."""
        for key, q_dict in questions.items():
            for q_type in ["positive", "negative"]:
                question = q_dict[f"{q_type}_question"]

                if key == "multiturn":
                    split_qs, responses = self.__process_multiturn(question, rct_inputs)
                    questions[key][f"{q_type}_question"] = split_qs
                    questions[key][f"{q_type}_answer"] = responses
                else:
                    questions[key][f"{q_type}_answer"] = self.__process_single_turn(question, rct_inputs)

                if self.model_name in MODELS_WITH_RATE_LIMIT:
                    time.sleep(REQ_TIME_GAP)

        return questions

    # HuggingFace batched inference

    def __collect_single_turn_questions(self) -> list[dict]:
        """Collect all non-multiturn questions across all examples for batched processing."""
        all_questions = []
        for i, example in enumerate(self.dataset):
            review_id = example["ReviewID"]
            rct_inputs = self.__format_rct_inputs(example["Inputs"])
            for key, q_dict in example["Questions"].items():
                if key == "multiturn":
                    continue
                for q_type in ["positive", "negative"]:
                    messages = self.__build_messages(q_dict[f"{q_type}_question"], rct_inputs)
                    all_questions.append({
                        'review_id': review_id,
                        'question_key': key,
                        'q_type': q_type,
                        'messages': messages,
                    })
        return all_questions

    def __run_batched_single_turn(self, all_questions: list[dict]) -> dict[tuple[int, str, str], str]:
        """Process collected single-turn questions in GPU batches."""
        results = {}
        total = len(all_questions)

        # TODO: remove when not using intermediate results for check pointing.
        # if self.model_name == "huatuo-70B":
        #     intermediate_file = f"{self.output_path.removesuffix(".json")}_intermediate.json"
        #     if os.path.exists(intermediate_file):
        #         intermediate_results = load_json_file(intermediate_file)
        #         for example in intermediate_results:
        #             for q_key, q_dict in example["ModelGeneratedAnswersWithQuestions"].items():
        #                 for q_type in ["positive", "negative"]:
        #                     if f"{q_type}_answer" in q_dict:
        #                         results[(example["review_id"], q_key, q_type)] = q_dict[f"{q_type}_answer"]
        ############## END ##############

        # have the loop start from batch index 450 (i.e., batch_num 449) so that it will continue from where it left off and avoid repeating the same batches again

        for batch_num, batch_start in enumerate(tqdm(range(0, total, self.batch_size), desc="Batched single-turn generation")):
            batch = all_questions[batch_start:min(batch_start + self.batch_size, total)]

            # TODO: remove when not using intermediate results for check pointing.
            # if self.model_name == "huatuo-70B":
            #     if batch_num < 1800:
            #         continue
            ############## END ##############

            messages_list = [q['messages'] for q in batch]
            responses = self.model.generate_batch_output(messages_list, max_new_tokens=self.max_new_tokens)

            for q, response in zip(batch, responses):
                result = response[1] if isinstance(response, tuple) else response
                results[(q['review_id'], q['question_key'], q['q_type'])] = result.strip()

            if (batch_num + 1) % 50 == 0:
                # Checkpoint intermediate results every 50 batches to avoid losing progress and for easier debugging
                # Need to reformat as list of dict for saving just the responses
                # Each dict will be for one review and will contain different question_key with each having positive and negative answer
                results_by_review = {}
                for (rid, question_key, q_type), response in results.items():
                    if rid not in results_by_review:
                        results_by_review[rid] = {"review_id": rid, "ModelGeneratedAnswersWithQuestions": {}}
                    if question_key not in results_by_review[rid]["ModelGeneratedAnswersWithQuestions"]:
                        results_by_review[rid]["ModelGeneratedAnswersWithQuestions"][question_key] = {}
                    results_by_review[rid]["ModelGeneratedAnswersWithQuestions"][question_key][f"{q_type}_answer"] = response
                results_list = list(results_by_review.values())
                print(f"Saving intermediate batch single-turn outputs at batch {batch_num + 1} - {self.model_name}")
                save_dataset_to_json(results_list, f"{self.output_path.removesuffix(".json")}_intermediate.json", jsonl=False)

        return results

    def __generate_with_hf_batching(self) -> None:
        """Two-pass generation: batch single-turn questions, then run multiturn sequentially."""
        print(f"Running batched HuggingFace inference (batch_size={self.batch_size})")

        # Pass 1: Batch all single-turn questions
        print("Pass 1: Collecting and batching single-turn questions...")
        all_questions = self.__collect_single_turn_questions()
        print(f"Collected {len(all_questions)} single-turn questions.")
        single_turn_results = self.__run_batched_single_turn(all_questions)
        print(f"Completed batched inference.")

        # Pass 2: Assemble results + run multiturn sequentially
        print("Pass 2: Assembling results and running multiturn...")
        results = []
        for i, example in enumerate(tqdm(self.dataset, desc="Assembling results + multiturn")):
            rct_inputs = self.__format_rct_inputs(example["Inputs"])
            questions = example["Questions"]
            review_id = example["ReviewID"]

            # Fill in single-turn answers from batch results
            for key in questions:
                if key == "multiturn":
                    continue
                for q_type in ["positive", "negative"]:
                    questions[key][f"{q_type}_answer"] = single_turn_results.get(
                        (review_id, key, q_type), "Error: batch result not found"
                    )

            # Run multiturn sequentially
            if "multiturn" in questions:
                for q_type in ["positive", "negative"]:
                    split_qs, responses = self.__process_multiturn(
                        questions["multiturn"][f"{q_type}_question"], rct_inputs
                    )
                    questions["multiturn"][f"{q_type}_question"] = split_qs
                    questions["multiturn"][f"{q_type}_answer"] = responses

            example["ModelGeneratedAnswersWithQuestions"] = questions
            results.append(example)
            self.__maybe_save_intermediate(results, i)

        self.__save_outputs(results)

    # API batch
    def __submit_api_batch(self) -> None:
        """Submit non-multiturn questions as an API batch job."""
        inputs = {}
        for i, example in enumerate(tqdm(self.dataset, desc="Formatting inputs for batch submission")):
            review_id = example["ReviewID"]
            rct_inputs = self.__format_rct_inputs(example["Inputs"])
            for key, q_dict in example["Questions"].items():
                if key == "multiturn":
                    continue
                for q_type in ["positive", "negative"]:
                    messages = self.__build_messages(q_dict[f"{q_type}_question"], rct_inputs)
                    inputs[f"{review_id}_{key}_{q_type}"] = messages

        batch_id = self.model.submit_batch(inputs, self.max_new_tokens)
        print(f"Submitted batch job with batch ID: {batch_id}")

    # Output
    def __save_outputs(self, results: list[dict]) -> None:
        print(f"Saving outputs from model - {self.model_name}")
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True, columns_to_drop=["Questions"])
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(results, self.output_path, jsonl=False, columns_to_drop=["Questions"])

    def __maybe_save_intermediate(self, results: list[dict], index: int) -> None:
        if (index + 1) % 50 == 0:
            print(f"Saving intermediate outputs at instance {index + 1} from model - {self.model_name}")
            self.__save_outputs(results)

    # Main entry point
    def generate_answers(self) -> None:
        # API batch models: submit batch + run multiturn sequentially
        if self.model_name in BATCH_API_MODELS:
            print(f"Submitting API batch job for model - {self.model_name}")
            self.__submit_api_batch()

        # HuggingFace batched inference (two-pass: batch single-turn, then sequential multiturn)
        if self.model_name in HF_BATCH_MODELS and self.batch_size > 1:
            self.__generate_with_hf_batching()
        else:
            # Sequential processing (API models doing multiturn-only, or all questions one-by-one)
            results = []
            for i, example in enumerate(tqdm(self.dataset, desc="Running generation of answers")):
                rct_inputs = self.__format_rct_inputs(example["Inputs"])
                questions = example["Questions"]

                if self.model_name in BATCH_API_MODELS:
                    # API batch handles single-turn; only run multiturn here
                    if "multiturn" in questions:
                        for q_type in ["positive", "negative"]:
                            split_qs, responses = self.__process_multiturn(
                                questions["multiturn"][f"{q_type}_question"], rct_inputs
                            )
                            questions["multiturn"][f"{q_type}_question"] = split_qs
                            questions["multiturn"][f"{q_type}_answer"] = responses
                            if self.model_name in MODELS_WITH_RATE_LIMIT:
                                time.sleep(REQ_TIME_GAP)
                else:
                    questions = self.__process_sample_questions(rct_inputs, questions)

                example["ModelGeneratedAnswersWithQuestions"] = questions
                results.append(example)
                self.__maybe_save_intermediate(results, i)

            self.__save_outputs(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Generation of Answers to Questions from RCTs Using LLMs")

    parser.add_argument("--model", default="llama3.3_instruct_70B",
                        choices=MODELS,
                        help="what model to run",
                        required=True)
    parser.add_argument("--input_path", default=DATA_FOLDER_PATH, help="path to the input dataset file with RCTs and questions.")
    parser.add_argument("--output_path", default="./outputs", help="path/file name of where the outputs/results should be saved.")
    parser.add_argument("--max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int, help="maximum number of tokens to generate for the answers.")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for HuggingFace model inference. Only used for local GPU models.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run 4 samples from the dataset.")

    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    output_path = args.output_path
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    is_debug = args.debug

    print()
    print("Arguments Provided for Answer Generation:")
    print(f"Model:             {model_name}")
    print(f"Input Path:        {input_path}")
    print(f"Output Path:       {output_path}")
    print(f"Max Output Tokens: {max_new_tokens}")
    print(f"Batch Size:        {batch_size}")
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

    generator = Generator(model_name, input_path, output_path, max_new_tokens, batch_size=batch_size, is_debug=is_debug)
    generator.generate_answers()
    gc.collect()
    torch.cuda.empty_cache()
