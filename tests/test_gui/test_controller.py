"""Tests for the Controller class in the controller module."""

import threading

from unittest.mock import Mock, MagicMock
from unittest import TestCase
from DollyChat.controller import Controller


class TestController(TestCase):
    """ Tests for basic controller functionality. """
    def __init__(self, *args, **kwargs):
        super(TestController, self).__init__(*args, **kwargs)
        self.test_controller = Controller(Mock(), Mock())

    def test_controller_init(self):
        assert self.test_controller.model is not None
        assert self.test_controller.view is not None

    def test_load_model_only_when_model_loaded_true(self):
        self.test_controller.model.model_loaded = False
        self.test_controller._load_model = Mock()
        self.test_controller.load_model()

        self.test_controller.model.model_loaded = True
        self.test_controller.load_model()
        self.test_controller._load_model.assert_called_once()


class TestLoadingThread(TestCase):
    """ Tests for the loading thread in the controller module. """
    def __init__(self, *args, **kwargs):
        super(TestLoadingThread, self).__init__(*args, **kwargs)
        self.test_controller = Controller(Mock(), Mock())
        self.load_event = threading.Event()

    def load_model_side_effect(self):
        """ Helper function to slow down the thread. """
        self.load_event.wait(0.5)

    def _reset_controller(self):
        del self.test_controller
        self.test_controller = Controller(Mock(), Mock())
        self.load_event = threading.Event()

    def test_load_model_starts_thread(self):
        """ Test that the load_model function starts a thread properly. """
        self.test_controller.model = MagicMock()
        self.test_controller.model.load_model.side_effect = self.load_model_side_effect
        self.test_controller.model.model_loaded = False
        self.test_controller.view.data_label.text = "false"
        self.test_controller.load_model()

        assert self.test_controller.lmm_loading_thread is not None
        assert self.test_controller.lmm_loading_thread.is_alive() is True

        # Represents the end of the thread
        self.test_controller.lmm_loading_thread.join()

        assert self.test_controller.lmm_loading_thread.is_alive() is False

        self._reset_controller()

    def test_load_model_changes_label(self):
        """ Test that the loading thread changes the label properly. """
        self.test_controller.model = MagicMock()
        self.test_controller.model.load_model.side_effect = self.load_model_side_effect
        self.test_controller.model.model_loaded = False
        self.test_controller.view.data_label.text = "false"
        self.test_controller.load_model()

        assert self.test_controller.view.data_label.text == "Loading model..."

        self.load_event.set()
        self.test_controller.lmm_loading_thread.join()

        assert self.test_controller.view.data_label.text == "Model loaded. Proceed."

        self._reset_controller()

    def test_load_model_calls_model_load(self):
        """ Test that the loading thread calls the model load function. """
        self.test_controller.model.model_loaded = False
        self.test_controller.model.load_model = Mock()
        self.test_controller.view.data_label.text = "false"
        self.test_controller.load_model()

        # Represents the end of the thread
        self.test_controller.lmm_loading_thread.join()

        self.test_controller.model.load_model.assert_called_once()

        self._reset_controller()
