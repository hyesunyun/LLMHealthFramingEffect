from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from models.model_utils import set_global_seed

SEED = 42

class Qwen3Thinking(Model):
    def __init__(self, model_size: str = "4B") -> None:
        super().__init__()
        set_seed(SEED)
        set_global_seed(SEED)
        
        if model_size == "4B":
            self.model_name = "Qwen/Qwen3-4B-Thinking-2507"
        elif model_size == "30B":
            self.model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
        logging.basicConfig(level=logging.ERROR)
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 262144
        
    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        ) # bfloat16 based on config.json

        # print model's dtype and device
        print(f"Model's dtype: {model.dtype}")
        print(f"Model's device: {model.device}")
        print(f"Model's device map: {model.hf_device_map}")
        print()
        
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.95) -> tuple[str, str]:
        """
        This method generates the output given the input. Uses chat template for input.

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation. default to 0.6 which are the recommended value from config file
        :param top_p: top_p for generation. default to 0.95 which are the recommended value from config file
        
        :return output of the model
        """
        try:
            message = [
                {"role": "user", "content": input},
            ]
            model_inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            
            do_sample = True if temperature > 0 else False

            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs, 
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
            return "{'error': f'Error: {e}'}", ""

