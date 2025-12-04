from .model import Model
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

SEED = 42

class GPT5(Model):
    def __init__(self, model_type: str = "5.1") -> None:
        super().__init__()
        if model_type == "5.1":
            self.model_name = "gpt-5.1-2025-11-13"
        elif model_type == "mini":
            self.model_name = "gpt-5-mini-2025-08-07"
        elif model_type == "nano":
            self.model_name = "gpt-5-nano-2025-08-07"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_context_length(self) -> int:
        return 400,000
    
    def generate_output(self, input: str, max_new_tokens: int, reasoning: str = "medium", verbosity: str = "medium", temperature: float = 1.0) -> tuple[str, str]:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param reasoning: controls how many reasoning tokens the model generates before producing a response. default to "none".
        :param verbosity: verbosity level for generation. default to "low".
        :param temperature: temperature for generation. default to 1.0. This can only be changed when reasoning is "none".

        :return output of the model
        """
        response = None
        try:
            if reasoning == "none":
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "input_text",
                            "text": input
                            }
                        ]
                        }
                    ],
                    reasoning={ "effort": reasoning },
                    temperature=temperature,
                    max_output_tokens=max_new_tokens,
                )
            else:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "input_text",
                            "text": input
                            }
                        ]
                        }
                    ],
                    reasoning={ "effort": reasoning },
                    text={ "verbosity": verbosity },
                )
            
        except Exception as e:
            logging.error(e)
        
        if response is None:
            return '{"error": "OpenAI GPT API call failed."}', ""
        else:
            return response.output_text, "" # No separate reasoning content returned