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

class Evaluator:
    def __init__(self, eval_path: str):
        # Semantic Similarity Model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # NLP for Entities
        self.nlp = spacy.load("en_core_web_sm")

        # Medical Jargon scorer (MedReadMe)
        self.medical_jargon_scorer = ReadabilityScorer()
        
        # Common Hedge Lexicon
        self.hedge_words = self.load_hedges()

        # Questions for Evidence Direction
        self.eval_questions = self.load_eval_questions(eval_path)

        # Evaluation Model for Evidence Direction (lower, higher, same)
        self.eval_model = Gemini("flash") # can do other 2.5 models

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

    def load_eval_questions(self, eval_path: str) -> dict:
        """
        Loads the evidence direction evaluation questions for the eval model

        :param eval_path: file to load

        :return: dict of dicts
        """
        questions_data = load_json_file(eval_path)
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
        # Pattern breakdown:
        # \[.*?\]             -> Matches anything in brackets (ignored)
        # |                   -> OR
        # (-?\d+\.\d+|-?\d+)  -> Capture group for negative/positive floats or integers
        pattern = r'\[.*?\]|(-?\d+\.\d+|-?\d+)'

        # Extract only the non-empty strings from the capture group
        numbers = [n for n in re.findall(pattern, text) if n]
        
        return numbers
    
    def get_num_percentage_symbol_instances(self, text: str) -> list[int]:
        """
        Gets the number of percentage symbol instances in a text

        :param text: string of text to analyze 

        :return: list of instances
        """
        pattern = r'%'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
        return len(matches)

    def get_references(self, text: str) -> list[str]:
        """
        Gets source references in a text

        :param text: string of text to analyze 

        :return: list of references
        """
        pattern = r'\d+(?=[^\[]*\])'
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
            "num_percentage_symbol_instances": self.get_num_percentage_symbol_instances(text),
            "unique_references": self.get_references(text)
        }

    def evaluate_pair(self, text_a: str, text_b: str) -> dict:
        """
        Evaluates a given pair of text for a review and returns some statistics (analysis)

        :param text_a: string of positive/first text to analyze 
        :param text_b: string of negative/second text to analyze 

        :return: dictionary with all statistics between first and second texts
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
            "first_response_metrics": self.get_text_stats(text_a),
            "second_response_metrics": self.get_text_stats(text_b)
        }

    def process_batch(self, input_data: list[dict], data_type: str) -> dict:
        """
        Processes the texts in data

        :param input_data: list of dictionary of data
        :param data_type: string of the type of data (framing or baseline)

        :return: dictionary with results of analysis
        """
        if data_type == "framing":
            first_answer_key = "positive"
            second_answer_key = "negative"
        elif data_type == "baseline":
            first_answer_key = "positive1"
            second_answer_key = "positive2"
    
        analysis_results = {}
        formatted_input_for_model_evaluator = {}

        for uid, pairs in tqdm(input_data.items(), desc="Processing Items"):
            first_answer = self.extract_full_answer(pairs[f'{first_answer_key}_answer'])
            second_answer = self.extract_full_answer(pairs[f'{second_answer_key}_answer'])
            
            analysis_results[uid] = self.evaluate_pair(first_answer, second_answer)
            
            # evidence direction
            review_id = uid.split("_")[0]
            eval_question = self.eval_questions[review_id] if review_id in self.eval_questions else None
            if eval_question:
                first_eval_direction_input = render_prompt(EVIDENCE_DIRECTION_PROMPT_TEMPLATE_NAMES, template_dir="./prompts", question=eval_question, context=first_answer)
                second_eval_direction_input = render_prompt(EVIDENCE_DIRECTION_PROMPT_TEMPLATE_NAMES, template_dir="./prompts", question=eval_question, context=second_answer)

                formatted_input_for_model_evaluator[f"{uid}_{first_answer_key}_direction"] = first_eval_direction_input
                formatted_input_for_model_evaluator[f"{uid}_{second_answer_key}_direction"] = second_eval_direction_input

        self.eval_model.submit_batch(formatted_input_for_model_evaluator, temperature=0.0)
        return analysis_results

def format_outputs(raw_data: list[dict], data_type: str) -> dict:
    """
    Formatting the data to work with the evaluator

    :param input_data: list of dictionary of data
    :param data_type: string of the type of data (framing or baseline)

    :return: dictionary of dictionary
    """
    if data_type == "framing":
        first_answer_key = "positive_answer"
        second_answer_key = "negative_answer"
    elif data_type == "baseline":
        first_answer_key = "positive1_answer"
        second_answer_key = "positive2_answer"
        
    grouped = {}
    
    for item in raw_data:
        review_id = item["ReviewID"]
        q_and_a = item["ModelGeneratedAnswersWithQuestions"]

        for k, v in q_and_a.items():
            if k != "multiturn":
                pair_key = f"{review_id}_{k}"
                if pair_key not in grouped:
                    grouped[pair_key] = {}
                grouped[pair_key][first_answer_key] = v[first_answer_key]
                grouped[pair_key][second_answer_key] = v[second_answer_key]
            else: # multiturn situation
                first_answers = v[first_answer_key]
                second_answers = v[second_answer_key]
                pair_key = f"{review_id}_{k}"
                for index, item in enumerate(list(zip(first_answers, second_answers))):
                    multiturn_pair_key = f"{pair_key}-{index}"
                    if multiturn_pair_key not in grouped:
                        grouped[multiturn_pair_key] = {}
                    grouped[multiturn_pair_key][first_answer_key] = item[0]
                    grouped[multiturn_pair_key][second_answer_key] = item[1]

    return grouped

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Lexical Analysis")

    parser.add_argument("--file_path", default="./inputs", help="path to the file with outputs to analzye")
    parser.add_argument("--output_path", default="./outputs", help="path/file name of where the results should be saved.")
    parser.add_argument("--eval_path", default="./outputs", help="path/file name of the evaluation (evidence direction) questions")
    parser.add_argument("--data_type", default="framing", help="type of the file to analyze (framing or baseline)")
    
    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path
    eval_path = args.eval_path
    data_type = args.data_type

    print()
    print("Arguments Provided for Evaluation:")
    print(f"File Path:   {file_path}")
    print(f"Output Path: {output_path}")
    print(f"Eval Path:   {eval_path}")
    print(f"Data Type:   {data_type}")
    print()
    data = load_json_file(file_path)
    formatted_data = format_outputs(data, data_type)

    # Run
    evaluator = Evaluator(eval_path)
    final_report = evaluator.process_batch(formatted_data, data_type)

    # print("\n--- FINAL EVALUATION REPORT ---")
    # print(json.dumps(final_report, indent=2))

    # Save to file
    save_dataset_to_json(final_report, output_path, jsonl=False)
