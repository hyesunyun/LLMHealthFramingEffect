# constants.py
from models.gpt5 import GPT5
from models.claude import Claude
from models.llama3 import Llama3
from models.qwen3 import Qwen3
from models.qwen3_thinking import Qwen3Thinking
from models.huatuo import Huatuo
from models.tacc_api import TACC

SEED = 42
MODELS_WITH_RATE_LIMIT = ["claude_4.5_sonnet", "api-llama3.3", "api-llama4"]
REQ_TIME_GAP = 10
MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", 
          "llama3.3_instruct_70B", "api-llama3.3", "api-llama4", 
          "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B",
          "huatuo-7B", "huatuo-8B", "huatuo-70B"]
REASONING_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "qwen3_thinking-4B", "qwen3_thinking-30B", "huatuo-7B", "huatuo-8B", "huatuo-70B"]
BATCH_API_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet"]
HF_BATCH_MODELS = [
    "llama3.3_instruct_70B",
    "qwen3-4B", "qwen3-30B",
    "qwen3_thinking-4B", "qwen3_thinking-30B",
    "huatuo-7B", "huatuo-8B", "huatuo-70B",
]
MODEL_CLASS_MAPPING = {
        "gpt-5.1": GPT5,
        "gpt5-mini": GPT5,
        "gpt5-nano": GPT5,
        "claude_4.5_sonnet": Claude,
        "llama3.3_instruct_70B": Llama3,
        "qwen3-4B": Qwen3,
        "qwen3-30B": Qwen3,
        "qwen3_thinking-4B": Qwen3Thinking,
        "qwen3_thinking-30B": Qwen3Thinking,
        "huatuo-7B": Huatuo,
        "huatuo-8B": Huatuo,
        "huatuo-70B": Huatuo,
        "api-llama3.3": TACC, 
        "api-llama4": TACC
    }