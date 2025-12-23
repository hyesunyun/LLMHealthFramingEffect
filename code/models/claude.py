from .model import Model
import anthropic
import os
import logging
from dotenv import load_dotenv

class Claude(Model):
    def __init__(self) -> None:
        super().__init__()
        
        self.model_name = "claude-sonnet-4-5-20250929"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_context_length(self) -> int:
        return 200000

    def generate_output(self, messages: list[dict], max_new_tokens: int) -> str:
        """
        This method generates the output given the input

        :param messages: messages with input for the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        completion = None
        try:
            completion = self.client.messages.create(
                model=self.model_name, # 200,000 tokens (https://docs.anthropic.com/en/docs/about-claude/models/overview)
                max_tokens=max_new_tokens,
                messages=messages,
            )
        except Exception as e:
            logging.error(e)
                
        if completion is None:
            return "Error: Anthropic Claude API call failed."
        else:
            return completion.content[0].text