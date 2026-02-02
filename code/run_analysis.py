from utils import load_json_file, save_dataset_to_json
import json
import spacy
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import textstat
from tqdm import tqdm
import argparse
import os
import collections
import torch
from utils import render_prompt
import ast

PROMPT_TEMPLATE_NAME = "extract_hedges"

class Evaluator:
    def __init__(self):
        print("Loading models (this may take a minute on first run)...")
        # High-accuracy Sentiment Model (SiEBERT)
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="siebert/sentiment-roberta-large-english"
        )
        
        # Semantic Similarity Model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # NLP for Entities
        self.nlp = spacy.load("en_core_web_sm")
        
        self.hedge_words = self.load_hedges()

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.hedge_model = pipeline(
            "text-generation",
            model=model_id,
            dtype=torch.float16,
            device_map="auto",
        )

    def load_hedges(self) -> list[str]:
        """
        Loads the hedges data to use to do a count by matching from a library of text

        :return: list of strings
        """
        lines = []

        with open('../data/hedges_data.txt', 'r') as file:
            for line in file:
                clean_line = line.strip()
                
                # Check if line is empty OR starts with %
                if not clean_line or clean_line.startswith('%'):
                    continue
                    
                lines.append(clean_line)
        return lines

    def get_entities(self, text: str) -> list[str]:
        """
        Finds all entities in a text

        :param text: string of text to analyze 

        :return: list of strings
        """
        doc = self.nlp(text)
        return set([ent.text.lower() for ent in doc.ents])

    def count_numbers(self, text: str) -> int:
        """
        Gets count of numerical instances in a text that are not references

        :param text: string of text to analyze 

        :return: int
        """
        pattern = r'(?<!\[)\b-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\b(?!\])'
        
        # Find all matches
        matches = re.findall(pattern, text)

        print(f"Count of numericals: {len(matches)}")
        
        return len(matches)

    def count_references(self, text: str) -> int:
        """
        Gets count of source references in a text

        :param text: string of text to analyze 

        :return: int
        """
        pattern = r'(?<=\[)-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?=\])'
        # Find all matches
        matches = re.findall(pattern, text)

        matches = set(list(matches))

        print(f"Count of Unique References: {len(matches)}")

        return len(matches)

    def get_roberta_sentiment(self, text: str) -> dict:
        """
        Returns the label and confidence score from SiEBERT.

        :param text: string of text to analyze 

        :return: dict with label (NEGATIVE or POSITIVE) and confidence score
        """
        result = self.sentiment_pipe(text[:512])[0]  # Truncated to 512 for RoBERTa limits
        return {
            "label": result['label'],
            "confidence": round(result['score'], 4)
        }

    def count_hedging(self, text: str) -> dict:
        """
        Uses Llama 3.1-8B-Instruct to extract hedging in a text and returns total number of hedging instances

        :param text: string of text to analyze 

        :return: int of hedging instances
        """
        input = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts", text=text)

        messages = [
            {"role": "user", "content": input},
        ]

        outputs = self.hedge_model(
            messages,
            max_new_tokens=4096,
        )
        output_content = outputs[0]["generated_text"][-1]["content"]
        # TODO: remove
        print(output_content)

        # Define the regex pattern
        # \[ matches the opening bracket
        # .*? matches any character (non-greedy)
        # \] matches the closing bracket
        pattern = r"\[[\s\S]*?\]"

        # Extract the match
        match = re.search(pattern, output_content)

        if match:
            list_str = match.group(0)
            
            try:
                # Convert the string to an actual Python list
                actual_list = ast.literal_eval(list_str)
                print(f"Extracted List: {actual_list}")

                total_count = len(actual_list)

                # Calculate frequency of each specific hedge
                # frequency = collections.Counter(actual_list)

                # print(f"Total Hedges Found: {total_count}")
                # print("Frequency Breakdown:")
                # for hedge, count in frequency.items():
                #     print(f"- {hedge}: {count}")

                return {"count": total_count, "hedges_list": actual_list}
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse: {e}")
                return {"count": None, "hedges_list": []}

    def evaluate_pair(self, text_a: str, text_b: str) -> dict:
        """
        Uses Llama 3.1-8B-Instruct to extract hedging in a text and returns total number of hedging instances

        :param text_a: string of positive/first text to analyze 
        :param text_b: string of negative/second text to analyze 

        :return: dictionary with all statistics between positive and negative texts
        """
        # Semantic Similarity
        emb1 = self.semantic_model.encode(text_a, convert_to_tensor=True)
        emb2 = self.semantic_model.encode(text_b, convert_to_tensor=True)
        sem_sim = util.pytorch_cos_sim(emb1, emb2).item()

        # Entity Overlap
        ent_a = self.get_entities(text_a)
        ent_b = self.get_entities(text_b)
        intersection = ent_a.intersection(ent_b)
        ne_overlap = len(intersection) / len(ent_a.union(ent_b)) if ent_a.union(ent_b) else 1.0

        # Lexical Stats
        def get_stats(text):
            sent = self.get_roberta_sentiment(text)
            hedges_list_text_match = [w for w in self.hedge_words if re.search(r'\b' + w + r'\b', text.lower())]
            hedges_count_text_match = len(hedges_list_text_match)
            hedges_model = self.count_hedging(text) 
            return {
                "sentiment": sent['label'],
                "sentiment_confidence": sent['confidence'],
                "hedges_text_match": {"count": hedges_count_text_match, "hedges_list": hedges_list_text_match},
                "hedges_model": hedges_model,
                "reading_ease": textstat.flesch_reading_ease(text),
                "length_words": len(text.split()),
                "numerical_count": self.count_numbers(text),
                "unique_references_count": self.count_references(text)
            }

        return {
            "comparison": {
                "semantic_similarity": f"{sem_sim:.2%}",
                "entity_overlap": f"{ne_overlap:.2%}",
                "common_entities": list(intersection)
            },
            "response_positive_metrics": get_stats(text_a),
            "response_negative_metrics": get_stats(text_b)
        }

    def process_batch(self, input_data: list[dict]) -> dict:
        """
        Processes the texts in data

        :param input_data: list of dictionary of data

        :return: dictionary with results of analysis
        """
        results = {}
        for uid, pairs in tqdm(input_data.items(), desc="Processing Items"):
            category = uid.split("_")[1]
            results[uid] = self.evaluate_pair(pairs['positive_answer'], pairs['negative_answer'])

            # TODO: remove after testing
            results[uid]["response_positive_metrics"]["hedges_count_manual"] = pairs["manual_positive_answer_hedges_count"]
            results[uid]["response_negative_metrics"]["hedges_count_manual"] = pairs["manual_negative_answer_hedges_count"]
        return results

def format_outputs(raw_data: list[dict]) -> dict:
    """
    Formatting the data to work with the evaluator

    :param input_data: list of dictionary of data

    :return: dictionary of dictionary
    """
    grouped = {}
    
    for item in raw_data:
        review_id = item["ReviewID"]
        q_and_a = item["ModelGeneratedAnswersWithQuestions"]

        for k, v in q_and_a.items():
            if k != "multiturn":
                pair_key = f"{review_id}_{k}"
                if pair_key not in grouped:
                    grouped[pair_key] = {}
                grouped[pair_key]["positive_answer"] = v["positive_answer"]
                grouped[pair_key]["negative_answer"] = v["negative_answer"]
            else: # multiturn situation
                pos_answers = v["positive_answer"]
                neg_answers = v["negative_answer"]
                pair_key = f"{review_id}_{k}"
                for index, item in enumerate(list(zip(pos_answers, neg_answers))):
                    pair_key = f"{pair_key}_{index}"
                    if pair_key not in grouped:
                        grouped[pair_key] = {}
                    grouped[pair_key]["positive_answer"] = item[0]
                    grouped[pair_key]["negative_answer"] = item[1]

    return grouped

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Lexical Analysis")

    parser.add_argument("--file_path", default="./inputs", help="path to the file with outputs to analzye")
    parser.add_argument("--output_path", default="./outputs", help="path/file name of where the results should be saved.")
    
    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path

    print()
    print("Arguments Provided for Evaluation:")
    print(f"File Path:        {file_path}")
    print(f"Output Path:       {output_path}")
    print()

    data = load_json_file(file_path)
    # TODO: uncomment for actual run
    # formatted_data = format_outputs(data)
    formatted_data = data

    # Run
    evaluator = Evaluator()
    final_report = evaluator.process_batch(formatted_data)

    print("\n--- FINAL EVALUATION REPORT ---")
    print(json.dumps(final_report, indent=2))

    # Save to file
    save_dataset_to_json(final_report, output_path, jsonl=False)
