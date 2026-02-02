from .model import Model
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
import json

SEED = 42
BASE_URL = "https://ai.tejas.tacc.utexas.edu/v1"

class TACC(Model):
    def __init__(self, model_type: str = "3.3") -> None:
        super().__init__()
        if model_type == "3.3":
            self.model_name = "Meta-Llama-3.3-70B-Instruct"
        elif model_type == "4":
            self.model_name = "Llama-4-Maverick-17B-128E-Instruct"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        api_key = os.getenv("TACC_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    def get_context_length(self) -> int:
        return 16000
    
    def generate_output(
        self, 
        messages: list[dict], 
        max_new_tokens: int
    ) -> str:
        """
        This method generates the output given the input

        :param messages: messages with input for the model
        :param max_new_tokens: maximum number of tokens to generate (this isn't really used)

        :return output of the model
        """
        response = None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )

        except Exception as e:
            logging.error(e)
        
        if response is None:
            return f"TACC AI API call failed."
        else:
            return response.choices[0].message.content.strip()