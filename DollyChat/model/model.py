import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from .instruct_pipeline import InstructionTextGenerationPipeline
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain


PATH = r"E:\Models\dolly-v2-3b"


class Model:
    def __init__(self, model_type: str = 'local', model_path=PATH):
        logging.basicConfig(
            filename='model.log',
            filemode='a+',
            format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
            datefmt='%d/%m/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.data = ''
        self.model_type = model_type.lower()
        self.model_path = model_path
        self._generator_model = None

        if self.model_type == 'local':
            self._generator_model = LocalDolly(self.model_path)
        elif self.model_type == 'remote':
            self._generator_model = RemoteDolly(self.model_path)

    def generate_response(self, prompt: str):
        return self._generator_model.generate_response(prompt)

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data


class LocalDolly:
    def __init__(self, model_path: str = PATH):
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
        self.logger.info('Model loaded')

        if on_gpu:
            self.model.to(self.device)
        self.logger.info("Model loaded successfully.")

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


class RemoteDolly:
    def __init__(self):
        self.model = None
