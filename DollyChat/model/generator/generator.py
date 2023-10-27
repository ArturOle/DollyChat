import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from model_zoo.dolly_v2_3b.instruct_pipeline import InstructionTextGenerationPipeline
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

PATH = r"./model_zoo/dolly-v2-3b"


class Generator:

    def __init__(self):
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
            PATH,
            padding_side="left",
            max_length=512,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        print('Model loaded')

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

    def generate_response(self, prompt, context=None, return_token_count=False):
        if prompt and context:
            response = self.generate_response_with_context(prompt, context)

            if return_token_count:
                return response, len(self.tokenizer.encode(response))
            else:
                return response

        elif prompt:
            response = self.generate_response_without_context(prompt)

            if return_token_count:
                return response, len(self.tokenizer.encode(response))
            else:
                return response

        else:
            raise AttributeError

    def generate_response_without_context(self, prompt: str):

        template = self.prompt_template

        llm_chain = LLMChain(
            llm=self.pipeline,
            prompt=template
        )
        self.logger.info("Generating response...")
        return llm_chain.predict(
            instruction=prompt
        ).lstrip()

    def generate_response_with_context(self, prompt, context):

        template = self.prompt_with_context_template

        llm_chain = LLMChain(
            llm=self.pipeline,
            prompt=template
        )

        self.logger.info("Generating response...")
        return llm_chain.predict(
            instruction=prompt,
            context=context
        ).lstrip()


if __name__ == "__main__":
    generator = Generator()
    while True:
        prompt = input("Enter prompt: ")
        print(generator.generate_response(prompt))
