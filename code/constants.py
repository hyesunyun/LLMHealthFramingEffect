# constants.py
from models.gpt5 import GPT5
from models.claude import Claude
from models.llama3 import Llama3
from models.deepseek import DeepSeek
from models.qwen3 import Qwen3
from models.qwen3_thinking import Qwen3Thinking

SEED = 42
MODELS_WITH_RATE_LIMIT = ["claude_4.5_sonnet"]
REQ_TIME_GAP = 5
MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "claude_4.5_sonnet", 
          "llama3.3_instruct_70B", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", 
          "qwen3-4B", "qwen3-30B", "qwen3_thinking-4B", "qwen3_thinking-30B"]
REASONING_MODELS = ["gpt-5.1", "gpt5-mini", "gpt5-nano", "deepseek_distill-qwen32B", "deepseek_distill-llama70B", "qwen3_thinking-4B", "qwen3_thinking-30B"]
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
    }