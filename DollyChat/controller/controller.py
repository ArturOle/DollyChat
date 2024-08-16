"""
Controller module for the MVC architecture. Responsible for handling the
communication between the model and the view.
"""
import logging
import threading


class Controller:
    """ Controller class for the MVC architecture. """
    def __init__(self, model, view):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        self.view = view
        self.model = model
        self.generating_flag = False
        self.response = None
        self.lmm_loading_thread = None
        self.llm_generating_thread = None

    def load_model(self):
        """ Load the model in a separate thread if not already loaded. """
        if self.model.model_loaded_flag is False:
            self._load_model()

    def unload_model(self):
        """ Unload the model. """
        self.model.unload_model()

    def _load_model(self):
        self.view.chat_screen.data_label.text = "Loading model..."
        self.lmm_loading_thread = threading.Thread(
            target=self._load_model_in_separate_thread
        )
        self.logger.info("Loading thread started")
        self.lmm_loading_thread.start()

    def _load_model_in_separate_thread(self):
        try:
            self.model.load_model()

        except TypeError:
            pass

        self.model.model_loaded_flag = True
        self.logger.info("Model loaded")
        self.view.chat_screen.data_label.text = "Model loaded. Proceed."

    def change_model(self, model_name):
        """ Change the model to the specified model if not already loaded. """
        if model_name == self.model.model_type:
            logging.info(f"Model {model_name} already loaded.")
            return
        self.unload_model()
        self.model.model_type = model_name
        self.load_model()

    def _generate_in_separate_thead(self, data):
        self.generating_flag = True
        self.response = self.model.generate_response(data)
        self.view.chat_screen.data_label.text = self.response
        self.generating_flag = False

    def update_data(self, data):
        """ Updates the view components with data obtained from model. """
        self.model.set_data(data)
        self.view.chat_screen.data_text = self.model.get_data()
        self.view.chat_screen.text_input.text = ""

    def generate_response(self, data):
        """
        Handles logic of generating response in separate thread to prevent
        GUI from freezing. Also handles the case where the model has not been
        loaded yet and the user tries to generate a response before generation
        of previous task is finished.
        """
        if self.model.model_loaded_flag is False:
            self.view.chat_screen.data_label.text = "Model not loaded yet. Please wait..."
        elif self.generating_flag is False:
            self.view.chat_screen.data_label.text = "Generating response..."
            self.llm_generating_thread = threading.Thread(
                target=self._generate_in_separate_thead,
                args=(data,)
            )
            self.llm_generating_thread.start()
            self.view.chat_screen.text_input.text = ""
        else:
            self.view.chat_screen.data_label.text = "Generating response. Please wait..."

    def get_available_models(self):
        """ Returns the available models. """
        return self.model.models_available.keys()

    def pop_up_ask_api_key(self):
        """ Pop up to ask for API key. """
        self.view.ask_api_key()

    def __del__(self):
        if self.lmm_loading_thread is not None and\
           self.lmm_loading_thread != threading.current_thread():

            self.lmm_loading_thread.join()

        if self.llm_generating_thread is not None and\
           self.lmm_loading_thread != threading.current_thread():

            self.llm_generating_thread.join()
