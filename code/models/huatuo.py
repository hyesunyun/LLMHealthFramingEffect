from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from models.model_utils import set_global_seed

SEED = 42

class Huatuo(Model):
    def __init__(self, model_type: str = "7B") -> None:
        super().__init__()
        set_seed(SEED)
        set_global_seed(SEED)
        
        if model_type == "7B":
            self.model_name = "FreedomIntelligence/HuatuoGPT-o1-7B"
        elif model_type == "8B":
            self.model_name = "FreedomIntelligence/HuatuoGPT-o1-8B"
        elif model_type == "70B":
            self.model_name = "FreedomIntelligence/HuatuoGPT-o1-70B"
        logging.basicConfig(level=logging.ERROR)
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 128000
        
    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", dtype="auto"
        )

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

    def generate_output(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.3, top_p: float = 0.8) -> tuple[str, str]:
        """
        This method generates the output given the input. Uses chat template for input.

        :param messages: messages with input for the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation. default to 0.3 which is the default value from config file
        :param top_p: top_p for generation. default to 0.8 which is the default value from config file
        
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

            # ## Thinking
            # [Reasoning process]

            # ## Final Response
            # [Output]
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            # print(f"Generated Output Text: {output_text}")
            try:
                thinking_start = output_text.index("## Thinking") + len("## Thinking")
                response_start = output_text.index("## Final Response")
                thinking_content = output_text[thinking_start:response_start].strip()
                content = output_text[response_start + len("## Final Response"):].strip()
            except ValueError as e:
                thinking_content = ""
                content = output_text.strip()
            
            return content, thinking_content
            
        except Exception as e:
            logging.error("[ERROR] %s", e)
            return f'{{"error": "Error: {e}"}}', ""

    def generate_batch_output(self, messages_list: list[list[dict]], max_new_tokens: int, temperature: float = 0.3, top_p: float = 0.8) -> list[tuple[str, str]]:
        """
        Generate outputs for a batch of message lists using left-padded batched inference.
        Parses out thinking content per sequence, returning only the final response.

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

                output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                # Parse "## Thinking" / "## Final Response" markers
                try:
                    thinking_start = output_text.index("## Thinking") + len("## Thinking")
                    response_start = output_text.index("## Final Response")
                    thinking_content = output_text[thinking_start:response_start].strip()
                    content = output_text[response_start + len("## Final Response"):].strip()
                except ValueError as e:
                    thinking_content = ""
                    content = output_text.strip()

                responses.append((thinking_content, content))
            self.tokenizer.padding_side = original_padding_side
            return responses

        except Exception as e:
            logging.error("[ERROR] %s", e)
            self.tokenizer.padding_side = original_padding_side
            return [f'{{"error": "Error: {e}"}}'] * len(messages_list)

