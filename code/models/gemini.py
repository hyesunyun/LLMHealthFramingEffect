from .model import Model
from google import genai
from google.genai import types
import os
import logging
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path

class Gemini(Model):
    def __init__(self, model_type: str = "2.5") -> None:
        super().__init__()
        
        if model_type == "2.5":
            self.model_name = "gemini-2.5-flash"
        else:
            self.model_name = "gemini-3-flash-preview"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=gemini_api_key)

    def get_context_length(self) -> int:
        return 1000000
    
    def _build_batch_requests(
        self,
        inputs: dict[str, str],
        temperature: float = 1.0,
    ) -> list[dict]:
        """
        Method for creating batch requests for Gemini Batch API
        
        :param inputs: dict[str, str] string of input text for each request
        :param temperature: temperature for generation. default to 1.0.

        :return list of requests for single model
        """
        requests = []

        for id, input in inputs.items():

            requests.append({
                "key": f"req-{id}",
                "request": {
                    "contents": [{"parts": [{"text": input}]}],
                    "generation_config": {"temperature": temperature}
                }
                
            })

        return requests
    
    def submit_batch(
        self,
        inputs: dict[str, str],
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> str:
        """
        Submits a batch job and returns the batch job name (batches/your-batch-id)

        :param inputs: dict[str, str] string of input text for each request
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature for generation. default to 1.0

        :return batch name
        """

        batch_requests = self._build_batch_requests(
            inputs,
            max_new_tokens,
            temperature
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
                "temperature": temperature,
                "created_at": timestamp,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Upload the file to the File API
            uploaded_file = self.client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(display_name=f"batch_requests_{timestamp}", mime_type='jsonl')
            )

            print(f"Uploaded batch input file: {uploaded_file.name}")

            # Create batch
            batch = self.client.batches.create(
                model=self.model_name,
                src=uploaded_file.name
            )

            print(f"Created batch: {batch.name}")

            return batch.name
        except Exception as e:
            logging.error(e)
            return None
    
    def generate_output(self, input: str, max_new_tokens: int, temperature: float = 1.0) -> tuple[str, str]:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: default temperature parameter for the model

        :return output of the model
        """
        response = None
        try:
            # default is dynamic thinking, meaning model will adjust the budget based on complexity of request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=input,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_new_tokens
                )
            )
        except Exception as e:
            logging.error(e)

        if response is None:
            return '{"error": "Gemini API call failed."}', ""
        else:
            return response.text, "" # No separate reasoning content returned