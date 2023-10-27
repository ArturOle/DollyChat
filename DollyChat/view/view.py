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

        self.label = Label(text="Data:")
        self.data_label = Label(text=self.data_text)
        self.text_input = TextInput()
        self.button = Button(text="Update Data")

        self.button.bind(on_press=self.update_data)

        self.add_widget(self.label)
        self.add_widget(self.data_label)
        self.add_widget(self.text_input)
        self.add_widget(self.button)

    def update_data(self, instance):
        data = self.text_input.text
        self.controller.update_data(data)
