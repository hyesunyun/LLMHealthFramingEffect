from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from dotenv import load_dotenv
import os
from models.model_utils import set_global_seed

SEED = 42

class Llama3(Model):
    def __init__(self) -> None:
        super().__init__()
        set_global_seed(SEED)
        set_seed(SEED)
        
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct"

        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        access_token=os.getenv("HUGGINGFACE_TOKEN")
        self.model = self.__load_model(access_token=access_token)
        self.tokenizer = self.__load_tokenizer(access_token=access_token)

    def get_context_length(self) -> int:
        return 128000
        
    def __load_model(self, access_token: str = None):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", dtype=torch.bfloat16, token=access_token,
        ) # bfloat16 based on config.json

        # print model's dtype and device
        print(f"Model's dtype: {model.dtype}")
        print(f"Model's device: {model.device}")
        print(f"Model's device map: {model.hf_device_map}")
        print()
        
        return model

    def __load_tokenizer(self, access_token: str = None):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def generate_output(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.9) -> str:
        """
        This method generates the output given the input. Uses chat template for input.

        :param messages: messages with input for the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation. default to 0.6 which are the recommended value from Meta
        :param top_p: top_p for generation. default to 0.9 which are the recommended value from Meta

        :return output of the model
        """
        try:
            model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            do_sample = True if temperature > 0 else False

            with torch.no_grad():
                result = self.model.generate(
                    model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                ) # do_sample = False and unset temperature and top_p (None) if we want deterministic outputs
                
            response = self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            
            return response

        except Exception as e:
            logging.error("[ERROR] %s", e)
            return f"Error: {e}"

    def generate_batch_output(self, messages_list: list[list[dict]], max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.9) -> list[str]:
        """
        Generate outputs for a batch of message lists using left-padded batched inference.

        :param messages_list: list of message lists, one per request
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation
        :param top_p: top_p for generation

        :return: list of response strings, one per input
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
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_length:]
            responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            self.tokenizer.padding_side = original_padding_side
            return responses

        except Exception as e:
            logging.error("[ERROR] %s", e)
            self.tokenizer.padding_side = original_padding_side
            return [f"Error: {e}"] * len(messages_list)
