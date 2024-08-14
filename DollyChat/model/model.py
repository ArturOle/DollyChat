""" Model class for the MVC architecture. """
import gc
import logging
import random
import time
import threading
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from .instruct_pipeline import InstructionTextGenerationPipeline
from langchain_nvidia_ai_endpoints import ChatNVIDIA


PATH = r"E:\Models\dolly-v2-3b"


class Model:
    """ Model class for the MVC architecture. """
    _generator_model = None

    def __init__(self, model_type: str = 'Remote Llama', model_path: str = PATH, api_path: str = 'api_key.txt'):
        logging.basicConfig(
            filename='model.log',
            filemode='a+',
            format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
            datefmt='%d/%m/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.models_available = {
            'Local Dolly': LocalDolly,
            'Fake Dolly': FakeDolly,
            'Remote Llama': RemoteLLaMA
        }
        self.api_path = api_path
        self.logger = logging.getLogger(__name__)
        self.data = ''
        self.model_type = ' '.join([mt.capitalize() for mt in model_type.split()])
        self.model_path = model_path
        self.model_loaded = False

    def unload_model(self):
        """ Resets the model. """
        del self._generator_model
        gc.collect()
        self._generator_model = None
        self.model_loaded = False

    def load_model(self):
        """ Loads the model. """
        if self.model_type == 'Remote Llama':
            try:
                with open(self.api_path, 'r', encoding='utf-8') as file:
                    api_key = file.read().strip()
            except FileNotFoundError:
                self.logger.error("API key file not found.")
                self.controller.pop_up_ask_api_key()
                return

        if self._generator_model is None:

            self._generator_model = self.models_available.get(self.model_type, None)
            if self._generator_model is None:
                raise KeyError(f"Model type {self.model_type} not available.")

        self._generator_model = self._generator_model()

    def generate_response(self, prompt: str):
        """ Generates a response using the model. """
        return self._generator_model.generate_response(
            prompt=prompt
        )


class LocalDolly:
    """ Logic for initializing and using Dolly LLLM model. """
    def __init__(self, model_path: str = PATH):
        print("I'm being innitiated!")
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

        on_gpu = torch.cuda.is_available()

        if on_gpu:
            self.logger.info("Generator is running on GPU with CUDA cores.")
            self.device = torch.device("cuda")
        else:
            self.logger.warning("Generator is running on CPUs.")
            self.device = torch.device("cpu")

        self.logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            max_length=512,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        if on_gpu:
            self.model.to(self.device)

        self.logger.info("Preparing text generation pipeline.")
        self.pipeline = InstructionTextGenerationPipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True
        )
        self.pipeline = HuggingFacePipeline(
            pipeline=self.pipeline
        )

        self.logger.info("Preparing templates.")
        self.prompt_template = PromptTemplate(
            input_variables=["instruction"],
            template="{instruction}"
        )
        self.prompt_with_context_template = PromptTemplate(
            input_variables=["instruction", "context"],
            template="{instruction}\n\nInput:\n{context}"
        )

        # if threading.current_thread() == threading.main_thread():
        #     exit()

    def generate_response(self, prompt: str):
        """ Generates a response using the model. """

        template = self.prompt_template

        llm_chain = LLMChain(
            llm=self.pipeline,
            prompt=template
        )
        self.logger.info("Generating response...")
        response = llm_chain.predict(
            instruction=prompt
        ).lstrip()
        self.logger.info(f"Response generated successfully:\n {response}")
        return response


class FakeDolly:
    """ Fake Dolly model for testing purposes. """
    excuses = [
        "I'm on a coffee break.",
        "I'm on a lunch break.",
        "I'm doing my laundry.",
        "I'm in a meeting.",
        "I'm doing my dishes.",
        "I'm cleaning my workspace.",
        "I'm walking my dog.",
        "I'm walking my cat.",
        "I'm walking my fish.",
        "I'm preparing my dinner.",
        "I'm preparing my breakfast.",
        "I'm preparing my coffee.",
        "SELECT excuse FROM SemiBuissnessRelatedExcusses ORDER BY RANDOM() LIMIT 1;"
    ]

    def __init__(self):
        self.model = None
        print("I'm bering innitiated!")

    def generate_response(self, prompt: str):
        """ Generates a fake response. """
        time.sleep(5)
        return f"""
            Sorry, I'm working remotely and {random.choice(self.excuses)}
            Please text me later.
        """


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
