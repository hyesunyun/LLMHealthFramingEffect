from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from models.model_utils import set_global_seed

SEED = 42

class Qwen3Thinking(Model):
    def __init__(self, model_type: str = "4B") -> None:
        super().__init__()
        set_seed(SEED)
        set_global_seed(SEED)
        
        if model_type == "4B":
            self.model_name = "Qwen/Qwen3-4B-Thinking-2507"
        elif model_type == "30B":
            self.model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
        logging.basicConfig(level=logging.ERROR)
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 262144
        
    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", dtype="auto"
        ) # bfloat16 based on config.json

        # print model's dtype and device
        print(f"Model's dtype: {model.dtype}")
        print(f"Model's device: {model.device}")
        print(f"Model's device map: {model.hf_device_map}")
        print()
        
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def generate_output(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.95) -> tuple[str, str]:
        """
        This method generates the output given the input. Uses chat template for input.

        :param messages: messages with input for the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation. default to 0.6 which are the recommended value from config file
        :param top_p: top_p for generation. default to 0.95 which are the recommended value from config file
        
        :return output of the model
        """
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            do_sample = True if temperature > 0 else False

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                ) # using default parameters from generatiion_config

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            return content, thinking_content
            
        except Exception as e:
            logging.error("[ERROR] %s", e)
            return f'{{"error": "Error: {e}"}}', ""

    def generate_batch_output(self, messages_list: list[list[dict]], max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.95) -> list[tuple[str, str]]:
        """
        Generate outputs for a batch of message lists using left-padded batched inference.
        Parses out thinking content per sequence, returning only the final content.

        :param messages_list: list of message lists, one per request
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation
        :param top_p: top_p for generation

        :return: list of tuples (thinking_content, content), one per input
        """
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"

            texts = [
                self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in messages_list
            ]

            model_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            do_sample = True if temperature > 0 else False

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )

            input_length = model_inputs.input_ids.shape[1]

            responses = []
            for i in range(generated_ids.shape[0]):
                output_ids = generated_ids[i, input_length:].tolist()

                # Remove padding tokens from output
                if self.tokenizer.pad_token_id is not None:
                    output_ids = [t for t in output_ids if t != self.tokenizer.pad_token_id]

                # Parse thinking content using token 151668 (</think>)
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                responses.append((thinking_content, content))

            self.tokenizer.padding_side = original_padding_side
            return responses

        except Exception as e:
            logging.error("[ERROR] %s", e)
            self.tokenizer.padding_side = original_padding_side
            return [f'{{"error": "Error: {e}"}}'] * len(messages_list)

