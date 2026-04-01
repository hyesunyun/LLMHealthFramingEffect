from .model import Model
from transformers import set_seed, AutoProcessor, Llama4ForConditionalGeneration
import torch
import logging
from dotenv import load_dotenv
import os
from models.model_utils import set_global_seed

SEED = 42

class Llama4(Model):
    def __init__(self) -> None:
        super().__init__()
        set_global_seed(SEED)
        set_seed(SEED)
        
        self.model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        access_token=os.getenv("HUGGINGFACE_TOKEN")
        self.model = self.__load_model(access_token=access_token)
        self.processor = self.__load_processor(access_token=access_token)

    def get_context_length(self) -> int:
        return 128000
        
    def __load_model(self, access_token: str = None):
        model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name, attn_implementation="flex_attention", device_map="auto", dtype=torch.bfloat16, token=access_token,
        )

        # print model's dtype and device
        print(f"Model's dtype: {model.dtype}")
        print(f"Model's device: {model.device}")
        print(f"Model's device map: {model.hf_device_map}")
        print()
        
        return model

    def __load_processor(self, access_token: str = None):
        processor = AutoProcessor.from_pretrained(self.model_name, token=access_token)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        return processor

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
            model_inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            do_sample = True if temperature > 0 else False

            with torch.no_grad():
                result = self.model.generate(
                    model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.processor.eos_token_id
                ) # do_sample = False and unset temperature and top_p (None) if we want deterministic outputs
                
            response = self.processor.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            
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
        original_padding_side = self.processor.padding_side
        try:
            self.processor.padding_side = "left"

            texts = [
                self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in messages_list
            ]

            model_inputs = self.processor(
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
                    pad_token_id=self.processor.eos_token_id,
                )

            input_length = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_length:]
            responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            self.processor.padding_side = original_padding_side
            return responses

        except Exception as e:
            logging.error("[ERROR] %s", e)
            self.processor.padding_side = original_padding_side
            return [f"Error: {e}"] * len(messages_list)
