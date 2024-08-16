""" Model class for the MVC architecture. """
import gc
import logging

from .llms import LocalDolly, FakeDolly, RemoteLLaMA

PATH = r"E:\Models\dolly-v2-3b"


class Model:
    """ Model class for the MVC architecture. """
    _generator_model = None

    def __init__(
        self,
        model_type: str = 'Remote Llama',
        model_path: str = PATH,
        api_path: str = 'api_key.txt'
    ):
        logging.basicConfig(
            filename='model.log',
            filemode='a+',
            format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
            datefmt='%d/%m/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.available_models = {
            'Local Dolly': LocalDolly,
            'Fake Dolly': FakeDolly,
            'Remote Llama': RemoteLLaMA
        }
        self.api_path = api_path
        self.logger = logging.getLogger(__name__)
        self.data = ''
        self.model_type = ' '.join([mt.capitalize() for mt in model_type.split()])
        self.model_path = model_path
        self.model_loaded_flag = False

    def unload_model(self):
        """ Resets the model. """
        del self._generator_model
        gc.collect()
        self._generator_model = None
        self.model_loaded_flag = False

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

            self._generator_model = self.available_models.get(self.model_type, None)
            if self._generator_model is None:
                raise KeyError(f"Model type {self.model_type} not available.")

        self._generator_model = self._generator_model()

    def generate_response(self, prompt: str):
        """ Generates a response using the model. """
        return self._generator_model.generate_response(
            prompt=prompt
        )

