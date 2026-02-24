from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def generate_output(self, messages: list[dict], max_new_tokens: int, temperature: float = 1.0) -> str:
        """
        This method must be overridden

        :abstract

        """
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """
        This method must be overridden

        :abstract

        """
        pass

    def generate_batch_output(self, messages_list: list[list[dict]], max_new_tokens: int) -> list[str]:
        """
        Generate outputs for a batch of message lists. Default implementation
        falls back to sequential single-item generation.

        Subclasses (HuggingFace models) should override this for GPU-batched inference.

        :param messages_list: list of message lists, one per request
        :param max_new_tokens: maximum number of tokens to generate

        :return: list of response strings, one per input
        """
        results = []
        for messages in messages_list:
            output = self.generate_output(messages, max_new_tokens)
            results.append(output)
        return results