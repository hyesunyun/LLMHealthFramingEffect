from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from models.model_utils import set_global_seed
from constants import SEED

# TODO: can consider batching/distributed inference to speed up generation.

class DeepSeek(Model):
    def __init__(self, model_type: str = "qwen32B") -> None:
        super().__init__()
        set_seed(SEED)
        set_global_seed(SEED)

        if model_type == "qwen32B":
            self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        elif model_type == "llama70B":
            self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        logging.basicConfig(level=logging.ERROR)
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 128000
        
    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", dtype=torch.bfloat16
        ) # bfloat16 based on config.json

        # print model's dtype and device
        print(f"Model's dtype: {model.dtype}")
        print(f"Model's device: {model.device}")
        print(f"Model's device map: {model.hf_device_map}")
        print()
        
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
            model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            do_sample=True if temperature > 0 else False
            with torch.no_grad():
                result = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, do_sample=do_sample, temperature=temperature, top_p=top_p)
            response = self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            
            # Need to remove the reasoning part from the response
            # </think>
            if "</think>" in response:
                context = response.split("</think>")
                response = context[-1].strip()
                thinking_content = context[0].strip()
            else:
                logging.warning("[WARNING] '</think>' token not found in response.")

            return response, thinking_content
        except Exception as e:
            logging.error("[ERROR] %s", e)
            return f'{{"error": "Error: {e}"}}', ""
