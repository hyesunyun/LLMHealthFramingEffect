from .model import Model
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
import json
from datetime import datetime

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
        return 400000

    def _build_batch_requests(
        self,
        all_messages: dict[str, list[dict]],
        max_new_tokens: int,
        reasoning: str = "medium",
        verbosity: str = "medium",
        temperature: float = 1.0,
    ) -> list[dict]:
        """
        Method for creating batch requests for OpenAI Batch API
        
        :param all_messages: dict[str, list[dict]] dict of formatted messages for each request
        :param max_new_tokens: maximum number of tokens to generate
        :param reasoning: controls how many reasoning tokens the model generates before producing a response. default to "none".
        :param verbosity: verbosity level for generation. default to "low".
        :param temperature: temperature for generation. default to 1.0. This can only be changed when reasoning is "none".

        :return list of requests for single model
        """
        requests = []

        for id, messages in all_messages.items():
            body = {
                "model": self.model_name,
                "input": messages,
            }

            if reasoning == "none":
                body.update({
                    "reasoning": {"effort": reasoning},
                    "temperature": temperature,
                    "max_output_tokens": max_new_tokens,
                })
            else:
                body.update({
                    "reasoning": {"effort": reasoning},
                    "text": {"verbosity": verbosity},
                })

            requests.append({
                "custom_id": f"req-{id}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            })

        return requests
    
    def submit_batch(
        self,
        all_messages: dict[str, list[dict]],
        max_new_tokens: int,
        reasoning: str = "medium",
        verbosity: str = "medium",
        temperature: float = 1.0,
        completion_window: str = "24h",
    ) -> str:
        """
        Submits a batch job and returns the batch_id

        :param all_messages: dict[str, list[dict]] dict of formatted messages for each request
        :param max_new_tokens: maximum number of tokens to generate
        :param reasoning: controls how many reasoning tokens the model generates before producing a response. default to "none".
        :param verbosity: verbosity level for generation. default to "low".
        :param temperature: temperature for generation. default to 1.0. This can only be changed when reasoning is "none".
        :param completion_window: time allowed for batch to complete

        :return batch_id
        """

        batch_requests = self._build_batch_requests(
            all_messages,
            max_new_tokens,
            reasoning,
            verbosity,
            temperature,
        )

        try:
            # Timestamped filename for record keeping
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_path = f"outputs/batch_requests/{self.model_name}_batch_{timestamp}.jsonl"

            # Write JSONL file locally
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")

            meta_path = f"outputs/batch_requests/{self.model_name}_batch_{timestamp}_meta.json"

            metadata = {
                "model": self.model_name,
                "num_requests": len(batch_requests),
                "reasoning": reasoning,
                "verbosity": verbosity,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "created_at": timestamp,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Upload file
            input_file = self.client.files.create(
                file=open(jsonl_path, "rb"),
                purpose="batch",
            )

            print(f"Uploaded batch input file: {input_file}")

            # Create batch
            batch = self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/responses",
                completion_window=completion_window,
            )

            print(f"Created batch: {batch}")

            return batch.id
        except Exception as e:
            logging.error(e)
            return None
    
    def generate_output(
        self, 
        messages: list[dict], 
        max_new_tokens: int, 
        reasoning: str = "medium", 
        verbosity: str = "medium", 
        temperature: float = 1.0
    ) -> tuple[str, str]:
        """
        This method generates the output given the input

        :param messages: messages with input for the model
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
                    input=messages,
                    reasoning={ "effort": reasoning },
                    temperature=temperature,
                    max_output_tokens=max_new_tokens,
                )
            else:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=messages,
                    reasoning={ "effort": reasoning },
                    text={ "verbosity": verbosity },
                )
            
        except Exception as e:
            logging.error(e)
        
        if response is None:
            return '{"error": "OpenAI GPT API call failed."}', ""
        else:
            return response.output_text, "" # No separate reasoning content returned
        