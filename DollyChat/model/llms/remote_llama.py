import logging


from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class RemoteLLaMA:
    """ WIP. Remote connection with LLaMA via NVIDIA API. """
    _api_key = None
    template = """
        <s>[INST] <<SYS>>
        You are expert in multiple fields. Keep the responses few sentences long.
        <</SYS>>

        {instruction} [/INST]
    """

    def __init__(self, api_path: str = 'api_key.txt'):
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

        self.logger.info("Connecting to NVIDIA API...")
        self.api_path = api_path
        self.client = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            api_key=self.api_key,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )

        self.logger.info("Connected to NVIDIA API.")

    @property
    def api_key(self):
        """ API key retrival property. """
        try:
            if self._api_key is None:
                with open(self.api_path, 'r', encoding='utf-8') as file:
                    self._api_key = file.read().strip()

            return self._api_key
        except FileNotFoundError:
            self.logger.error("API key file not found.")
            self._api_key = None

    def generate_response(self, prompt: str):
        """ Generates a response using the model. """
        prompt_template = PromptTemplate(
            input_variables=["instruction"],
            template=self.template,
        )
        prompt = [{
            "role": "user",
            "content": prompt_template.format(instruction=prompt)
        }]
        response = ""

        for chunk in self.client.stream(prompt):
            response += chunk.content

        return response


if __name__ == '__main__':
    model = RemoteLLaMA()
    print(model.generate_response("Tell me a story about those medival times!"))
