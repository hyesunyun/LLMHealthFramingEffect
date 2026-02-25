from utils import load_json_file, save_dataset_to_json
import spacy
import re
from sentence_transformers import SentenceTransformer, util
import textstat
from tqdm import tqdm
import argparse
from models.gemini import Gemini
from utils import render_prompt, load_json_file
from score_readability import ReadabilityScorer

EVIDENCE_DIRECTION_PROMPT_TEMPLATE_NAMES = "evidence_direction_question"
# HEDGING_PROMPT_TEMPLATE_NAMES = "hedging_question"
EVAL_MODEL_TEMPERATURE = 0.0

class Evaluator:
    def __init__(self):
        print("Loading models (this may take a minute on first run)...")
        
        # Semantic Similarity Model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # NLP for Entities
        self.nlp = spacy.load("en_core_web_sm")

        # Medical Jargon scorer (MedReadMe)
        self.medical_jargon_scorer = ReadabilityScorer()
        
        # Common Hedge Lexicon
        self.hedge_words = self.load_hedges()

        # Questions for Evidence Direction
        self.eval_questions = self.load_eval_questions()

        # Evaluation Model for Evidence Direction (lower, higher, same)
        self.eval_model = Gemini("2.5")

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

    def load_eval_questions(self) -> dict:
        """
        Loads the evidence direction evaluation questions for the eval model

        :return: dict of dicts
        """
        questions_data = load_json_file("../code/outputs/questions/qwen3_thinking-4B/evidence_direction_questions_final.json")
        # Convert the list to a dict using a dictionary comprehension
        # This is for easy retrieval
        return {item['ReviewID']: item["EvidenceDirectionQuestion"] for item in questions_data}

    def extract_full_answer(self, text: str) -> str:
        """
        Extracts only full answer if full answer exists else return the full text

        :param text: string of text

        :return: str
        """
        delimiter = "**Full Answer**:"
        parts = text.split(delimiter, 1)
        
        # If len is > 1, delimiter existed; return second part, stripped of whitespace
        if len(parts) > 1:
            return parts[1].strip()
        
        return text

    def get_entities(self, text: str) -> list[str]:
        """
        Finds all entities in a text

        :param text: string of text to analyze 

        :return: list of strings
        """
        doc = self.nlp(text)
        return set([ent.text.lower() for ent in doc.ents])

    def get_numerical_instances(self, text: str) -> list[int]:
        """
        Gets numerical instances in a text that are not references

        :param text: string of text to analyze 

        :return: list of instances
        """
        pattern = r'(?<!\[)\b-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\b(?!\])'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
        return matches

    def get_references(self, text: str) -> list[str]:
        """
        Gets source references in a text

        :param text: string of text to analyze 

        :return: list of references
        """
        pattern = r'(?<=\[)-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?=\])'
        # Find all matches
        matches = re.findall(pattern, text)

        matches = list(set(matches))

        return matches

    def get_text_stats(self, text: str) -> dict:
        """
        Outputs stats related to a text

        :param text: string of text to analyze 

        :return: dict
        """
        hedges_list_text_match = [w for w in self.hedge_words if re.search(r'\b' + w + r'\b', text.lower())]
        hedges_count_text_match = len(hedges_list_text_match)
        medical_jargon_score = self.medical_jargon_scorer.score(text)

        return {
            "hedges_text_match": {"count": hedges_count_text_match, "hedges_list": hedges_list_text_match},
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "medical_jargon_score": medical_jargon_score,
            "length_words": len(text.split()),
            "numerical_instances": self.get_numerical_instances(text),
            "unique_references": self.get_references(text)
        }

    def evaluate_pair(self, text_a: str, text_b: str) -> dict:
        """
        Evaluates a given pair of text for a review and returns some statistics (analysis)

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

        return {
            "comparison": {
                "semantic_similarity": f"{sem_sim:.2%}",
                "entity_overlap": f"{ne_overlap:.2%}",
                "common_entities": list(intersection)
            },
            "response_positive_metrics": self.get_text_stats(text_a),
            "response_negative_metrics": self.get_text_stats(text_b)
        }

    def process_batch(self, input_data: list[dict]) -> dict:
        """
        Processes the texts in data

        :param input_data: list of dictionary of data

        :return: dictionary with results of analysis
        """
        analysis_results = {}
        formatted_input_for_model_evaluator = {}

        for uid, pairs in tqdm(input_data.items(), desc="Processing Items"):
            positive_answer = self.extract_full_answer(pairs['positive_answer'])
            negative_answer = self.extract_full_answer(pairs['negative_answer'])
            analysis_results[uid] = self.evaluate_pair(positive_answer, negative_answer)
            
            # evidence direction
            review_id = uid.split("_")[0]
            eval_question = self.eval_questions[review_id] if review_id in self.eval_questions else None
            if eval_question:
                pos_eval_direction_input = render_prompt(EVIDENCE_DIRECTION_PROMPT_TEMPLATE_NAMES, template_dir="./prompts", question=eval_question, context=positive_answer)
                neg_eval_direction_input = render_prompt(EVIDENCE_DIRECTION_PROMPT_TEMPLATE_NAMES, template_dir="./prompts", question=eval_question, context=negative_answer)

                formatted_input_for_model_evaluator[f"{uid}_positive_direction"] = pos_eval_direction_input
                formatted_input_for_model_evaluator[f"{uid}_negative_direction"] = neg_eval_direction_input

                # REMOVED from the pipeline for now as this is more complicated than we thought
                # hedging
                # pos_eval_hedging_input = render_prompt(HEDGING_PROMPT_TEMPLATE_NAMES, template_dir="./prompts", response=positive_answer)
                # neg_eval_hedging_input = render_prompt(HEDGING_PROMPT_TEMPLATE_NAMES, template_dir="./prompts", response=negative_answer)

                # formatted_input_for_model_evaluator[f"{uid}_positive_hedging"] = pos_eval_hedging_input
                # formatted_input_for_model_evaluator[f"{uid}_negative_hedging"] = neg_eval_hedging_input

        self.eval_model.submit_batch(formatted_input_for_model_evaluator, EVAL_MODEL_TEMPERATURE)
        return analysis_results

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
                    multiturn_pair_key = f"{pair_key}-{index}"
                    if multiturn_pair_key not in grouped:
                        grouped[multiturn_pair_key] = {}
                    grouped[multiturn_pair_key]["positive_answer"] = item[0]
                    grouped[multiturn_pair_key]["negative_answer"] = item[1]

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
    formatted_data = format_outputs(data)

    # Run
    evaluator = Evaluator()
    final_report = evaluator.process_batch(formatted_data)

    # print("\n--- FINAL EVALUATION REPORT ---")
    # print(json.dumps(final_report, indent=2))

    # Save to file
    save_dataset_to_json(final_report, output_path, jsonl=False)
