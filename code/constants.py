# constants.py
from models.gpt5 import GPT5
from models.claude import Claude
from models.llama3 import Llama3
from models.deepseek import DeepSeek
from models.qwen3 import Qwen3
from models.qwen3_thinking import Qwen3Thinking
from models.gemini import Gemini
from models.huatuo import Huatuo
from models.tacc_api import TACC

SEED = 42
MODELS_WITH_RATE_LIMIT = ["claude_4.5_sonnet", "gemini-2.5", "gemini-3", "api_llama-3.3", "api_llama-4"]
REQ_TIME_GAP = 10
MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", 
          "llama3.3_instruct_70B", "api_llama-3.3", "api_llama-4", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", 
          "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B", "gemini-2.5", "gemini-3",
          "huatuo-7B", "huatuo-8B", "huatuo-70B"]
REASONING_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "deepseek_distill-qwen32B", 
                    "deepseek_distill-llama70B", "qwen3_thinking-4B", "qwen3_thinking-30B", 
                    "gemini-2.5", "gemini-3", "huatuo-7B", "huatuo-8B", "huatuo-70B"]
BATCH_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", "gemini-2.5", "gemini-3"]
MODEL_CLASS_MAPPING = {
        "gpt-5.1": GPT5,
        "gpt5-mini": GPT5,
        "gpt5-nano": GPT5,
        "claude_4.5_sonnet": Claude,
        "deepseek_distill-qwen32B": DeepSeek,
        "deepseek_distill-llama70B": DeepSeek,
        "llama3.3_instruct_70B": Llama3,
        "qwen3-4B": Qwen3,
        "qwen3-30B": Qwen3,
        "qwen3_thinking-4B": Qwen3Thinking,
        "qwen3_thinking-30B": Qwen3Thinking,
        "gemini-2.5": Gemini,
        "gemini-3": Gemini,
        "huatuo-7B": Huatuo,
        "huatuo-8B": Huatuo,
        "huatuo-70B": Huatuo,
        "api_llama-3.3": TACC, 
        "api_llama-4": TACC
    }