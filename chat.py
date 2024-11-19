import os

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from openai import OpenAI
import replicate
import os

load_dotenv(override = True)

class Chat(ABC):
    @abstractmethod
    def single_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Complete a single request"""
        pass

class OpenAIChat(Chat):
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        self.client = OpenAI()
        self.model_name = "gpt-4o-2024-08-06"

    def single_completion(self, system_prompt: str, user_prompt: str, max_retry=5) -> str:
        completion = self.client.chat.completions.create(
            model = self.model_name,
            temperature = 0,
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return completion.choices[0].message.content


class Llama3Chat(Chat):
    ACCEPTED_NAMES = ["meta-llama-3-8b-instruct", "meta-llama-3-70b-instruct"]
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name not in self.ACCEPTED_NAMES:
            raise ValueError("Model Name Error")
    def single_completion(self, system_prompt: str, user_prompt: str) -> str:
        model_input = {
            "prompt": user_prompt,
            "system_prompt": system_prompt,
            "temperature": 0,
            "max_new_tokens": 512,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        }
        if self.model_name == "meta-llama-3-8b-instruct":
            result = "".join(replicate.run("meta/meta-llama-3-8b-instruct",input = model_input))
        elif self.model_name == "meta-llama-3-70b-instruct":
            result = "".join(replicate.run("meta/meta-llama-3-70b-instruct", input = model_input))
        else:
            raise ValueError("Model Not supported")
        return result