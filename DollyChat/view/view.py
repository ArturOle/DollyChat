from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown


class View(BoxLayout):
    data_text = StringProperty()

    def __init__(self, controller, **kwargs):
        super(View, self).__init__(**kwargs)
        self.controller = controller
        self.orientation = 'vertical'
        self.build()

    def build(self):
        self.label = Label(text="Response:", size_hint_y=0.1)
        self.data_label = Label(
            text=self.data_text,
            size_hint_y=0.6,
            valign='middle',
            halign='center'
        )

        self.data_label.bind(size=self._update_text_size)

        self.text_input = TextInput(size_hint_y=0.2)
        self.text_input.bind(text=self._on_enter)

        self.add_widget(self.label)
        self.add_widget(self.data_label)
        self.add_widget(self.text_input)

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
            text="Generate Response",
            valign='middle',
            halign='center'
        )

        self.button.bind(
            on_press=self._update_data,
            size=self._update_text_size
        )
        self.h_box.add_widget(self.button)
        self.help_button = Button(
            text="Help",
            valign='middle',
            halign='center'
        )
        # self.help_button.bind(on_press=self.help)
        self.h_box.add_widget(self.help_button)

        self.add_widget(self.h_box)

    def _update_text_size(self, instance, value):
        instance.text_size = (instance.width, None)

    def _update_data(self, instance):
        data = self.text_input.text
        self.controller.generate_response(data)

    def _on_enter(self, instance, value):
        if self.text_input.text.endswith("\n"):
            self._update_data(instance)
            self.text_input.text = ""
