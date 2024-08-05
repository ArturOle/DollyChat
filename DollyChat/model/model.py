import torch
import logging
import random
import time
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer
from .instruct_pipeline import InstructionTextGenerationPipeline
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain


PATH = r"E:\Models\dolly-v2-3b"


class Model:
    _generator_model = None

    def __init__(self, model_type: str = 'fake', model_path: str = PATH):
        logging.basicConfig(
            filename='model.log',
            filemode='a+',
            format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
            datefmt='%d/%m/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.models_available = {
            'local': LocalDolly,
            'fake': FakeDolly,
            'remotellama': RemoteLLaMA
        }
        self.logger = logging.getLogger(__name__)
        self.data = ''
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.model_loaded = False

    @property
    def generator_model(self):
        if self._generator_model is None:
            try:
                self._generator_model = self.models_available[self.model_type]()

            except KeyError as err:
                raise ValueError(
                    f"""
                    Invalid model type: {self.model_type}.
                    Must be one of: {', '.join(self.models_available.keys())}
                    """
                ) from err
            self.model_loaded = True
        return self._generator_model

    def load_model(self):
        self.generator_model()
        self.model_loaded = True

    def generate_response(self, prompt: str):
        return self._generator_model.generate_response(prompt)

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data


class LocalDolly:
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

        if threading.current_thread() == threading.main_thread():
            exit()

    def generate_response(self, prompt: str):

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
    excuses = [
        # "I'm on a coffee break.",
        # "I'm on a lunch break.",
        # "I'm doing my laundry.",
        # "I'm in a meeting.",
        # "I'm doing my dishes.",
        # "I'm cleaning my workspace.",
        # "I'm walking my dog.",
        # "I'm walking my cat.",
        # "I'm walking my fish.",
        # "I'm preparing my dinner.",
        # "I'm preparing my breakfast.",
        # "I'm preparing my coffee.",
        "SELECT excuse FROM SemiBuissnessRelatedExcusses ORDER BY RANDOM() LIMIT 1;"
    ]

    def __init__(self):
        self.model = None
        print("I'm bering innitiated!")

    def generate_response(self, _):
        time.sleep(5)
        return f"Sorry, I'm working remotely and {random.choice(self.excuses)} Please text me later."


class RemoteLLaMA:
    def __init__(self):
        self._api_key = None
        self.model = None

    def set_api_key(self, path):
        self._api_key = open(path).read().strip()
