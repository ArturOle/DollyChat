""" Test cases for the Model classes. """

from unittest.mock import Mock, MagicMock
from unittest import TestCase
from DollyChat.model import Model


class TestModel(TestCase):
    """ Tests for the Model class in the model module. """
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.test_model = Model("fake")

    def _reset_model(self):
        self.test_model = Model("fake")

    def test_model_init(self):
        assert self.test_model.model_type == "fake"

        self.test_model = Model("FAKE")
        assert self.test_model.model_type == "fake"

        self.test_model = Model("TotallyNotFake")
        assert self.test_model.model_type == "totallynotfake"

        self._reset_model()

    def test_generator_model(self):
        self.test_model.model_loaded = False
        self.test_model.models_available = {
            'fake': MagicMock()
        }

        self.test_model.generator_model
        assert self.test_model._generator_model is not None
        assert self.test_model.model_loaded is True

        self.test_model = Model("totallynotfake")
        self.test_model.models_available = {
            'fake': MagicMock()
        }
        with self.assertRaises(KeyError):
            self.test_model.generator_model

        self._reset_model()

    def test_load_model(self):
        self.test_model._generator_model = MagicMock()
        self.test_model.load_model()

        assert self.test_model.model_loaded is True
        self.test_model._generator_model.assert_called_once()

        self._reset_model()
