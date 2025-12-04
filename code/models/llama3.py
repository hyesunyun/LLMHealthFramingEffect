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

    def generate_output(self, input: str, max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.9) -> str:
        """
        This method generates the output given the input. Uses chat template for input.

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation. default to 0.6 which are the recommended value from Meta
        :param top_p: top_p for generation. default to 0.9 which are the recommended value from Meta

        :return output of the model
        """
        try:
            message = [
                {"role": "user", "content": input},
            ]
            model_inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            
            do_sample = True if temperature > 0 else False

            with torch.no_grad():
                result = self.model.generate(
                    model_inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                ) # do_sample = False and unset temperature and top_p (None) if we want deterministic outputs
                
            response = self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logging.error("[ERROR] %s", e)
            return f"Error: {e}"
