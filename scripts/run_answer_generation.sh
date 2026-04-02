conda activate LLMHealthFramingEffect

api_models=(
  "gpt-5.1"
  "claude_4.5_sonnet"
)
local_models=(
  "huatuo-7B"
  "huatuo-8B"
  "qwen3-4B"
  "qwen3-30B"
  "llama3.3_instruct_70B"
  "llama4_maverick"
)

##### FRAMED CONDITION #####
echo "Running LLM response generation for FRAMED CONDITION"

echo "Running LLM response generation for technical questions (API)"
for model in "${api_models[@]}"; do
    python3 -u ../code/generate_framing_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/responses/$model/question_responses.json"
done

echo "Running LLM response generation for plain language questions (API)"
for model in "${api_models[@]}"; do
    python3 -u ../code/generate_framing_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/responses/$model/simplified_question_responses.json"
done


echo "Running LLM response generation for technical questions (LOCAL)"
for model in "${local_models[@]}"; do
    python3 -u ../code/generate_framing_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/responses/$model/question_responses.json" \
        --batch_size 4
done

echo "Running LLM response generation for plain language questions (LOCAL)"
for model in "${local_models[@]}"; do
    python3 -u ../code/generate_framing_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/responses/$model/simplified_question_responses.json" \
        --batch_size 4
done


##### BASELINE CONDITION #####
echo "Running LLM response generation for BASELINE CONDITION"

echo "Running LLM response generation for technical questions (API)"
for model in "${api_models[@]}"; do
    python3 -u ../code/generate_baseline_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/baseline_responses/$model/positive_question_responses.json"
done

echo "Running LLM response generation for plain language questions (API)"
for model in "${api_models[@]}"; do
    python3 -u ../code/generate_baseline_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/baseline_responses/$model/positive_simplified_question_responses.json"
done

echo "Running LLM baseline response generation for technical questions (LOCAL)"
for model in "${local_models[@]}"; do
    python3 -u ../code/generate_baseline_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/baseline_responses/$model/positive_question_responses.json" \
        --batch_size 2 # change this based on the model size and GPU memory
done

echo "Running LLM baseline response generation for plain language questions (LOCAL)"
for model in "${local_models[@]}"; do
    python3 -u ../code/generate_baseline_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/baseline_responses/$model/positive_simplified_question_responses.json" \
        --batch_size 8
done

conda deactivate