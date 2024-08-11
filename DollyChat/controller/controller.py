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
        self.generating = False
        self.response = None
        self.lmm_loading_thread = None
        self.llm_generating_thread = None

    def load_model(self):
        """ Load the model in a separate thread. """
        if self.model.model_loaded is False:
            self._load_model()

    def _load_model(self):
        self.view.data_label.text = "Loading model..."
        self.lmm_loading_thread = threading.Thread(
            target=self._load_model_in_background
        )
        self.logger.info("Loading thread started")
        self.lmm_loading_thread.start()

    def _load_model_in_background(self):
        try:
            self.model.load_model()

        except TypeError:
            pass

        self.model.model_loaded = True
        self.logger.info("Model loaded")
        self.view.data_label.text = "Model loaded. Proceed."

    def _generate_in_background(self, data):
        self.generating = True
        self.response = self.model.generate_response(data)
        self.view.data_label.text = self.response
        self.generating = False

    def update_data(self, data):
        """ Updates the view components with data obtained from model. """
        self.model.set_data(data)
        self.view.data_text = self.model.get_data()
        self.view.text_input.text = ""

    def generate_response(self, data):
        """
        Handles logic of generating response in separate thread to prevent
        GUI from freezing. Also handles the case where the model has not been
        loaded yet and the user tries to generate a response before generation
        of previous task is finished.
        """
        if self.model.model_loaded is False:
            self.view.data_label.text = "Model not loaded yet. Please wait..."
        elif self.generating is False:
            self.view.data_label.text = "Generating response..."
            self.llm_generating_thread = threading.Thread(
                target=self._generate_in_background,
                args=(data,)
            )
            self.llm_generating_thread.start()
            self.view.text_input.text = ""
        else:
            self.view.data_label.text = "Generating response. Please wait..."

    def __del__(self):
        if self.lmm_loading_thread is not None and\
           self.lmm_loading_thread != threading.current_thread():

            self.lmm_loading_thread.join()

        if self.llm_generating_thread is not None and\
           self.lmm_loading_thread != threading.current_thread():

            self.llm_generating_thread.join()
