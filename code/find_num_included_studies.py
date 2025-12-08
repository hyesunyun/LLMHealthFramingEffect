import argparse
import os

import requests
from bs4 import BeautifulSoup

from utils import load_json_file, load_jsonl_file, save_dataset_to_json
from tqdm import tqdm
import time

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DELAY_BETWEEN_REQUESTS = 3  # seconds to wait between requests to be polite to the server

class WebScraper:

    def __init__(self, input_path: str, output_path: str, is_debug: bool = False) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.is_debug = is_debug

        self.dataset = None
        self.session = None
        self.__load_dataset()
        self.__init_session()

    def __load_dataset(self) -> None:
        """
        This method loads the dataset based on input path.

        :return dataset as a list of dictionaries
        """
        print("Loading the dataset...")
        if self.input_path.endswith(".jsonl"):
            dataset = load_jsonl_file(self.input_path)
        elif self.input_path.endswith(".json"):
            dataset = load_json_file(self.input_path)

        if self.is_debug:
            dataset = dataset[:5] # use only first 5 examples for debugging

        self.dataset = dataset

    def __init_session(self) -> None:
        # Initialize a Session
        s = requests.Session()

        # Add the User-Agent header to the session
        s.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
        })
        self.session = s

    def find_num_studies(self) -> None:
        results = []
        print("Starting to scrape number of included studies...")
        for example in tqdm(self.dataset):
            doi = example["DOI"]
            url = f"https://www.cochranelibrary.com/cdsr/doi/{doi}/references"
            print(url)
            response = self.session.get(url)

            if response.status_code != 200:
                print(f"[ERROR] Failed to retrieve page for DOI {doi}. Status code: {response.status_code}")
                example["NumIncludedStudies"] = None
                results.append(example)
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            references_section = soup.find("section", id="references", class_="references bibliographies")

            if not references_section:
                print(f"[WARNING] References section not found for DOI {doi}.")
                example["NumIncludedStudies"] = 0
                results.append(example)
                continue

            included_studies_divs = references_section.find_all("div", class_="bibliographies references_includedStudies")
            num_included_studies = len(included_studies_divs)

            example["NumIncludedStudies"] = num_included_studies
            results.append(example)

            # add a small delay to be polite to the server
            time.sleep(DELAY_BETWEEN_REQUESTS)

        print("Scraping completed. Saving results...")
        # save results to a json file
        if self.output_path.endswith(".jsonl"):
            save_dataset_to_json(results, self.output_path, jsonl=True)
        elif self.output_path.endswith(".json"):
            save_dataset_to_json(results, self.output_path)
        print(f"Results saved to {self.output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running the Finding of Number of Studies Included in Cochrane Reviews Using Web Scraping")

    parser.add_argument("--input_path", default="../data", help="path to the input file containing DOIs.", required=True),
    parser.add_argument("--output_path", default="./outputs", help="path with file name to save outputs.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will run 10 sampled data.")
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for Scraper:")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print(f"Is Debug: {is_debug}")
    print()
    
    scraper = WebScraper(input_path, output_path, is_debug)
    scraper.find_num_studies()