from pathlib import Path
from .model import Model
import anthropic
import os
import logging
from dotenv import load_dotenv
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import json
from datetime import datetime

class Claude(Model):
    def __init__(self) -> None:
        super().__init__()
        
        self.model_name = "claude-sonnet-4-5-20250929"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_context_length(self) -> int:
        return 200000
    
    def _build_batch_requests(
        self,
        all_messages: dict[str, list[dict]],
        max_new_tokens: int
    ) -> list[dict]:
        """
        Method for creating batch requests for OpenAI Batch API
        
        :param all_messages: dict[str, list[dict]] dict of formatted messages for each request
        :param max_new_tokens: maximum number of tokens to generate

        :return list of requests for single model
        """
        requests = []

        for id, messages in all_messages.items():
            requests.append(Request(
                custom_id=f"req-{id}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    messages=messages
                )
            ))

        return requests
    
    def submit_batch(
        self,
        all_messages: dict[str, list[dict]],
        max_new_tokens: int
    ) -> str:
        """
        Submits a batch job and returns the batch_id

        :param all_messages: dict[str, list[dict]] dict of formatted messages for each request
        :param max_new_tokens: maximum number of tokens to generate

        :return batch_id
        """

        batch_requests = self._build_batch_requests(
            all_messages,
            max_new_tokens
        )

        try:
            # Timestamped filename for record keeping
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            current_file = Path(__file__).resolve()
            target_dir = current_file.parent.parent / "outputs" / "batch_requests"
            target_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = target_dir / f"{self.model_name}_batch_{timestamp}.jsonl"

            # Write JSONL file locally
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")

            meta_path = target_dir / f"{self.model_name}_batch_{timestamp}_meta.json"

            metadata = {
                "model": self.model_name,
                "num_requests": len(batch_requests),
                "max_new_tokens": max_new_tokens,
                "created_at": timestamp
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Create batch
            batch = self.client.messages.batches.create(requests=batch_requests)

            print(f"Created batch: {batch}")

            return batch.id
        except Exception as e:
            logging.error(e)
            return None

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