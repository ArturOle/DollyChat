import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

PATH = r"E:\Models\dolly-v2-3b"


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
