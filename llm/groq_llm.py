import os
from groq import Groq
from .base_llm import BaseLLM

class GroqLLM(BaseLLM):

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL")

    def generate(self, system_prompt, user_prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.6,
            max_tokens=150,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content