from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen


class View:

    def __init__(self, controller):
        self.controller = controller
        self.screen_manager = self.build()

    def build(self):
        sm = ScreenManager()

        chat_screen = ChatScreen(name='chat')
        selection_screen = SelectionScreen(name='selection')
        #help_screen = HelpScreen(name='help')

        sm.add_widget(self.chat_screen)
        #sm.add_widget(self.selection_screen)
        #sm.add_widget(self.help_screen)

        return sm

class ChatScreen(Screen):
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

    def _go_to_selection(self, instance):
        self.controller.go_to_selection()

    def _go_to_help(self, instance):
        self.controller.go_to_help()


class SelectionScreen(Screen):
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

    def _go_to_chat(self, instance):
        self.controller.go_to_chat()


class HelpScreen(Screen):
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

    def _go_to_chat(self, instance):
        self.controller.go_to_chat()
