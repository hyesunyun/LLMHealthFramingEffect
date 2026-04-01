import argparse
from utils import load_json_file, load_jsonl_file, save_dataset_to_json
from tqdm import tqdm

def load_data(input_path: str) -> list[dict]:
    """
    This method loads the dataset based on input path.

    :params input_path: path to the input file

    :return dataset as a list of dictionaries
    """
    print("Loading the dataset...")
    if input_path.endswith(".jsonl"):
        dataset = load_jsonl_file(input_path)
    elif input_path.endswith(".json"):
        dataset = load_json_file(input_path)
    return dataset

def save_data(dataset: list, output_path: str) -> None:
    """
    This method saves the dataset to the specified output path.

    :params dataset: list of dictionaries to save
    :params output_path: path to the output file

    :return: None
    """
    print(f"Saving the dataset to {output_path}...")
    # save results to a json file
    if output_path.endswith(".jsonl"):
        save_dataset_to_json(dataset, output_path, jsonl=True)
    elif output_path.endswith(".json"):
        save_dataset_to_json(dataset, output_path)
    
def filter_reviews(input_path: str, output_path: str) -> None:
    """
    This method filters out reviews with mismatched number of studies.
    NumInputs (number of studies we have in the dataset) != NumIncludedStudies (number of studies included in the review)

    :params input_path: path to the input file
    :params output_path: path to the output file

    :return: None
    """
    # load the data
    dataset = load_data(input_path)
    print(f"Loaded data with rows: {len(dataset)}")

    filtered_data = []
    pbar = tqdm(dataset, desc="Running filtering reviews with mismatched studies")
    for _, example in enumerate(pbar):
        num_inputs = example["NumInputs"]
        num_included_studies = example["NumIncludedStudies"]

        if num_inputs is None or num_included_studies is None:
            print(f"Skipping example with missing NumInputs or NumIncludedStudies: {example['ReviewID']}")
            continue
        
        # also exclude reviews that have any input abstracts to be None (empty)
        inputs = example["Inputs"]
        if any(input["Abstract"] is None for input in inputs):
            continue

        # only include if the numbers match
        if num_inputs == num_included_studies:
            filtered_data.append(example)

    print(f"Filtered data with rows: {len(filtered_data)}")

    # save the filtered data
    save_data(filtered_data, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running the Filtering of Reviews (Remove Reviews with Mismatched Studies)")

    parser.add_argument("--input_path", default="../data", help="path to the input file containing NumInputs & NumIncludedStudies", required=True),
    parser.add_argument("--output_path", default="../data", help="path with file name to save filtered outputs.")
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    print("Arguments Provided:")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print()
    
    filter_reviews(input_path, output_path)