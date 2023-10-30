import asyncio

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.uix.textinput import TextInput


class View(BoxLayout):
    data_text = StringProperty()

    def __init__(self, controller, **kwargs):
        super(View, self).__init__(**kwargs)
        self.controller = controller
        self.orientation = 'vertical'
        self.build()

    def build(self):
        self.label = Label(text="Data:")
        self.data_label = Label(text=self.data_text)
        self.text_input = TextInput()
        self.button = Button(text="Generate Response")

        self.button.bind(on_press=self.update_data)

        self.add_widget(self.label)
        self.add_widget(self.data_label)
        self.add_widget(self.text_input)
        self.add_widget(self.button)

    def update_data(self, instance):
        data = self.text_input.text
        asyncio.run(self.controller.generate_response(data))
        response = self.controller.response
        self.data_text = response
