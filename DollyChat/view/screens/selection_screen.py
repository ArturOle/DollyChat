from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen
from kivy.uix.checkbox import CheckBox

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCheckBox:
    def __init__(self, model_name, controller):
        self.h_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.1
        )
        self.model_name = model_name
        self.controller = controller
        self.build()

    def build(self):
        self.label = Label(
            text=self.model_name
        )
        self.h_box.add_widget(self.label)
        self.checkbox = CheckBox(
            group='models',
            color=(1, 1, 1, 1),
            active=True
        )
        self.h_box.add_widget(self.checkbox)


class ModelCheckBoxGroup(BoxLayout):
    def __init__(self, available_models, controller):
        super(ModelCheckBoxGroup, self).__init__()
        self.orientation = 'vertical'
        self.size_hint_y = 0.6
        self.size_hint_x = 0.6
        self.available_models = available_models
        self.controller = controller
        self.checkboxes = {}
        self.build()

    def build(self):
        for model in self.available_models:
            self.checkboxes[model] = ModelCheckBox(model, self.controller)
            self.checkboxes[model].checkbox.bind(on_press=self._on_select)
            self.add_widget(self.checkboxes[model].h_box)

    def find_selected(self):
        for model, checkbox in self.checkboxes.items():
            if checkbox.checkbox.active:
                return model

    def set_selection(self, model):
        for checkbox in self.checkboxes.values():
            if checkbox.model_name == model:
                checkbox.checkbox.active = True
            else:
                checkbox.checkbox.active = False

    def _on_select(self, instance):
        for checkbox in self.checkboxes.values():
            if checkbox.checkbox != instance:
                checkbox.checkbox.active = False
            else:
                checkbox.checkbox.active = True
                logger.info(f"Selected model: {checkbox.model_name}")


class SelectionScreen(Screen):
    data_text = StringProperty()

    def __init__(self, controller, **kwargs):
        super(SelectionScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.controller = controller
        self.build()

    def build(self):
        self.label = Label(text="Select Model:", size_hint_y=0.1)
        self.layout.add_widget(self.label)
        available_models = self.controller.get_available_models()

        self.model_checkboxes = ModelCheckBoxGroup(
            available_models,
            self.controller
        )
        self.layout.add_widget(self.model_checkboxes)

        self.submit_button = Button(
            text="Submit",
            valign='middle',
            halign='center',
            size_hint_y=0.1,
            size_hint_x=0.4
        )
        self.submit_button.bind(on_press=self._submit)
        self.layout.add_widget(self.submit_button)

        self.h_box = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.1
        )
        self.model_select = Button(
            text="Select Model",
            valign='middle',
            halign='center'
        )
        self.model_select.bind(size=self._update_text_size)
        self.h_box.add_widget(self.model_select)
        self.button = Button(
            text="Back to Chat",
            valign='middle',
            halign='center'
        )

        self.button.bind(
            on_press=self._go_to_chat,
            size=self._update_text_size
        )
        self.h_box.add_widget(self.button)
        self.help_button = Button(
            text="Help",
            valign='middle',
            halign='center'
        )

        self.help_button.bind(on_press=self._go_to_help)
        self.h_box.add_widget(self.help_button)

        self.layout.add_widget(self.h_box)
        self.add_widget(self.layout)

    def _update_text_size(self, instance, value):
        instance.text_size = (instance.width, None)

    def _update_data(self, instance):
        data = self.text_input.text
        self.controller.generate_response(data)

    def _on_enter(self, instance, value):
        if self.text_input.text.endswith("\n"):
            self._update_data(instance)
            self.text_input.text = ""

    def _go_to_chat(self, instance):
        self.manager.current = 'chat'
        self.model_checkboxes.set_selection(self.controller.model.model_type)

    def _go_to_help(self, instance):
        self.manager.current = 'help'
        self.model_checkboxes.set_selection(self.controller.model.model_type)

    def _submit(self, instance):
        selected_model = self.model_checkboxes.find_selected()
        self.controller.change_model(selected_model)
        self.manager.current = 'chat'
