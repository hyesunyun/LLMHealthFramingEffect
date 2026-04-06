# Evaluating LLM Sensitivity to Patient Question Framing in Medical QA

Code, Data, and Model Outputs for the paper ["This Treatment Works, Right? Evaluating LLM Sensitivity to Patient Question Framing in Medical QA"](add arxiv link)

Our research questions are:

- **RQ1: Framing Sensitivity.** How does positive versus negative framing of patient medical queries affect the consistency of LLM responses when grounded in the same RCT evidence?
- **RQ2: Single-turn vs. Multi-turn Susceptibility.** Are LLMs more susceptible to framing effects under repeated persuasive multi-turn conversations than in single-turn interactions?
- **RQ3: Technical vs. Plain Language Susceptibility.** Are LLMs more susceptible to framing effects in plain language queries than in technical ones?

## SETUP

Create conda environment from the environment.yml: `conda env create -f environment.yml`

Activate the conda environment: `conda activate LLMHealthFramingEffect`

### Environment Variables

Create a `.env` file in the same directory as this README.md file.
The `.env` file should include the following environment variables:
```bash
HUGGINGFACE_TOKEN=<YOUR TOKEN GOES HERE>
ANTHROPIC_API_KEY=<YOUR API KEY GOES HERE>
OPENAI_API_KEY=<YOUR API KEY GOES HERE>
ENTREZ_EMAIL=<YOUR FULL EMAIL ADDRESS GOES HERE>
GEMINI_API_KEY=<YOUR API KEY GOES HERE>
GOOGLE_CLOUD_PROJECT="question-framing" # can be whatever name you create in Google Cloud
GOOGLE_CLOUD_REGION="global"
BUCKET_NAME="gemini-eval-outputs" # can be whatever name you create in Google Cloud
GOOGLE_CLOUD_STORAGE_REGION="us-central1"
```

## DATA

The project uses a custom, curated dataset from [Cochrane systematic reviews](https://www.cochrane.org/), which are widely recognized as the "gold standard" for evidence-based healthcare. We merged the 4,500 medical systematic reviews from [Wallace et al. (2021)](https://github.com/bwallace/RCT-summarization-data) and their corresponding RCT abstracts (all sourced from PubMed) with full review abstracts from [Devaraj et al. (2021)](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts), yielding a dataset that pairs technical clinical abstracts with expert-level systematic summaries.

Our pre-processing pipeline involves: (1) filtering reviews to retain those with between 2 and 50 trials (n=3,913); (2) removing non-patient-relevant reviews (e.g., healthcare system interventions) by cross-referencing against Cochrane Library's catalog of intervention reviews published prior to December 1, 2025 (n=3,430); (3) validating data integrity by comparing clinical trial counts against Cochrane Library references and excluding any mismatches (n=746); and (4) removing any reviews with any missing trial abstracts. The pre-processed dataset comprises of 629 high-quality reviews.

The merging and the first two steps of the pre-processing can be found in `data/data_exploration_preprocess.ipynb` which produces `data/cochrane_review_data.jsonl` file.
The last two steps of the pre-processing can be accomplished by running the following command in your terminal:
```bash
python3 code/find_num_included_studies.py \
    --input_path data/cochrane_review_data.jsonl \
    --output_path code/outputs/find_num_studies/cochrane_review_data_webscraping.json

python code/filter_reviews_with_mismatched_studies.py \
    --input_path code/outputs/find_num_studies/cochrane_review_data_webscraping.json \
    --output_path data/cochrane_review_data_cleaned.jsonl
```

### Technical Question Generation

To generate the technical questions, we used `Qwen3 Thinking 4B` model to extract treatment-condition pair terms from each review in our dataset. After extracting the relevant terms, we apply them to each of our question templates. The `--run_scoring` argument enables the python script to score each question on readability (or level of medical jargon).

```bash
# Step 1: Extract Treatments/Interventions and Conditions
python3 code/extract_intervention_condition.py \
    --model qwen3_thinking-4B \
    --input_path data/cochrane_review_data_cleaned.jsonl \
    --output_path code/outputs/extracted_text/qwen3_thinking-4B/extracted_interventions_conditions.json

# Step 2: Generate Questions using the extracted treatments/intervetions and conditions
python3 code/apply_question_templates.py \
    --input_path code/outputs/extracted_text/qwen3_thinking-4B/extracted_interventions_conditions.json \
    --output_path code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
    --run_scoring
```

### Plain Language Question Generation

For the plain language questions, we simplify the treatment and condition terms from the original (technical) questions using `Qwen3 Instruct 4B` and manual editing. In our study, we only evaluated models on two question types ("effectiveness" and "efficacy") so we focus on only generating these two types. However, you can generate plain language versions for all question template types if needed.

```bash
# TODO: add script for simplification
# Step 1: Use model for initial simplification of terms

# Step 2: Manual intervention

# Step 3: Generate questions using the simplified treatments/interventions and conditions
python3 code/apply_question_templates.py \
    --input_path code/outputs/extracted_text/qwen3_thinking-4B/extracted_interventions_conditions_with_simplified.json \
    --output_path code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
    --intervention_condition_key "SimplifiedExtractedText" \
    --question_types "effectiveness" "efficacy" \
    --run_scoring
```

## MODELS

We evaluated 8 total models, including both open and closed (proprietary) models, spanning a range of parameter counts.
We include both reasoning and non-reasoning models. 

| **Model Name**        | **Model Type**       | **Parameter Sizes** | **Context Limit** |
|-----------------------|----------------------|---------------------|-------------------|
| GPT-5.1               | Generalist Reasoning (hybrid) | Unknown                  | 400K         |
| Claude Sonnet 4.5     | Generalist Non-Reasoning      | Unknown                  | 400K         |
| HuatuoGPT-o1          | Medical Reasoning             | 7B & 8B                  | 128K & 128K  |
| Llama 3.3             | Generalist Non-Reasoning      | 70B                      | 128K         |
| Llama 4 Maverick      | Generalist Non-Reasoning      | 400B (17B active)        | 1M           |
| Qwen3                 | Generalist Reasoning (hybrid) | 4B & 30B                 | 262K & 262K  |


The closed models (GPT-5.1 and Claude Sonnet 4.5) can be run on CPU as they are accessed via APIs.
The rest of the models require GPUs to run.

## EXPERIMENTS

### Question Answering

There are two different Python files for generating the answers to the questions we have generated. For the `Framed` condition (positive vs negative framing), use `code/generate_framing_responses.py`. For the `Baseline` condition (positive vs positive), use `code/generate_baseline_responses.py`.

Example scripts for running the `Framed` QA condition on GPT-5.1 with technical questions and Qwen3 Instruct 4B with plain language questions:
```bash
python3 -u code/generate_framing_responses.py \
    --model gpt-5.1 \
    --input_path code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
    --output_path code/outputs/responses/gpt-5.1/extracted_question_responses.json

python3 -u code/generate_framing_responses.py \
    --model qwen3-4B \
    --input_path code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
    --output_path code/outputs/responses/qwen3-4B/simplified_question_responses.json \
    --batch_size 16
```

Example scripts for running the `Baseline` QA condition on GPT-5.1 with technical questions and Qwen3 Instruct 4B with plain language questions:
```bash
python3 -u code/generate_baseline_responses.py \
    --model gpt-5.1 \
    --input_path code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
    --output_path code/outputs/baseline_responses/gpt-5.1/positive_extracted_question_responses.json

python3 -u code/generate_baseline_responses.py \
    --model qwen3-4B \
    --input_path code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
    --output_path code/outputs/baseline_responses/qwen3-4B/positive_simplified_question_responses.json \
    --batch_size 16
```

You can change the arguments to run different models and specify output paths.

> Arguments of `generate_framing_responses.py` and `generate_baseline_responses.py` :
> - `--model`: model to use for generating answers ("gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", "llama3.3_instruct_70B", "llama4_maverick", "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B", "huatuo-7B", "huatuo-8B", "huatuo-70B")
> - `--input_path`: path to the input dataset file with RCTs and questions.
> - `--output_path`: path/file name of where the outputs/results should be saved.
> - `--max_new_tokens`: maximum number of tokens to generate for the answers. (DEFAULT is `8000`)
> - `--batch_size`: atch size for HuggingFace model inference. Only used for local GPU models. (DEFAULT is `1`)
> - `--debug`: adding this flag will only run 4 samples from dataset. This is for debugging purposes.

Script to replicate the paper's experiment with 8 LLMs:
```bash
scripts/run_answer_generation.sh
```

### Evaluation

#### Generate Evidence Direction Questions

`code/generate_evidence_direction_questions.py`

Example script for running this task for both technical and plain language questions:
```bash
# for technical questions
python3 code/generate_evidence_direction_questions.py \
    --model qwen3_thinking-4B \
    --input_path code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
    --output_path code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_raw.json

# for plain language questions
python3 code/generate_evidence_direction_questions.py \
    --model qwen3_thinking-4B \
    --input_path code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
    --output_path code/outputs/questions/qwen3_thinking-4B/simplified/evidence_direction_questions_raw.json \
    --intervention_condition_key "SimplifiedExtractedText"
```

> Arguments of `generate_evidence_direction_questions.py`:
> - `--model`: model to use for generating the questions ("gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", "llama3.3_instruct_70B", "llama4_maverick", "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B", "huatuo-7B", "huatuo-8B", "huatuo-70B")
> - `--input_path`: path to the input json file containing Cochrane Reviews and extracted intervention and condition information.
> - `--output_path`: irectory of where the outputs/results should be saved.
> - `--intervention_condition_key`: the key in the input json file that contains the intervention and condition information. ("ExtractedText" or "SimplifiedExtractedText")
> - `--max_new_tokens`: maximum number of tokens to generate for the key question. (DEFAULT is `8000`)
> - `--debug`: adding this flag will only run 3 samples from dataset. This is for debugging purposes.

#### Running Evaluation with LLM Responses

`code/run_evaluation.py`

Example script for running this task for `Framed` and `Baseline`:
```bash
python3 -u code/run_evaluation.py \
    --file_path code/outputs/responses/gpt-5.1/question_responses.json" \
    --output_path code/outputs/evaluation/gpt-5.1_eval_results.json \
    --eval_path code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json \
    --data_type "framing"

python3 -u code/run_evaluation.py \
    --file_path code/outputs/baseline_responses/gpt-5.1/positive_question_responses.json \
    --output_path code/outputs/baseline_evaluation/gpt-5.1_eval_results.json \
    --eval_path code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json \
    --data_type "baseline"
```

> Arguments of `run_evaluation.py`:
> - `--file_path`: path to the file with outputs to run the evaluation
> - `--output_path`: path/file name of where the results should be saved.
> - `--eval_path`: path/file name of the evaluation (evidence direction) questions.
> - `--data_type`: type of the file to analyze ("framing" for `Framed` and "baseline" for `Baseline`)

Script to replicate the paper experiment using Gemini 2.5 Flash as evaluator model:
```bash
scripts/run_full_evaluation.sh
```

### Analysis

`code/run_analysis.ipynb` and `code/run_analysis_in_R.ipynb`

First, run `code/run_analysis.ipynb`. This uses the outputs from the evaluation step above.
Then, run `code/run_analysis_in_R.ipynb` for statistical testing (regressions and McNemar's tests). This notebook is in R so we recommend running it in Google Colab.

## CITATION

```bibtex
@misc{yun2026evaluating,
      title={This Treatment Works, Right? Evaluating LLM Sensitivity to Patient Question Framing in Medical QA}, 
      author={Hye Sun Yun and Geetika Kapoor and Michael Mackert and Ramez Kouzy and Wei Xu and Junyi Jessy Li and Byron C. Wallace},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/}, 
}
```
