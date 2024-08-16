import logging

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView

from .screens import ChatScreen
from .screens import SelectionScreen
from .screens import HelpScreen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class View(ScreenManager):

    def __init__(self, controller):
        super(View, self).__init__()
        self.controller = controller
        self.screen_manager = None
        self.chat_screen = None
        self.selection_screen = None
        self.help_screen = None
        self._build()

    def _build(self):
        self.screen_manager = ScreenManager()

        self.chat_screen = ChatScreen(
            self.controller,
            name='chat'
        )
        self.selection_screen = SelectionScreen(
            self.controller,
            name='selection'
        )
        self.help_screen = HelpScreen(
            self.controller,
            name='help'
        )

        self.add_widget(self.chat_screen)
        self.add_widget(self.selection_screen)
        self.add_widget(self.help_screen)
        self.current = 'chat'

    def show_file_chooser(self):
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView()
        select_button = Button(text="Select")

        popup = Popup(
            title="Select a file",
            content=content,
            size_hint=(0.9, 0.9)
        )

        select_button.bind(
            on_release=lambda x: self.file_selected(filechooser.path, popup)
        )

        content.add_widget(filechooser)
        content.add_widget(select_button)

        popup.open()

    def show_info_gathering(self):
        content = BoxLayout(orientation='vertical')
        text_input = TextInput(hint_text="Enter some information")
        submit_button = Button(text="Submit")

        popup = Popup(
            title="Enter Information",
            content=content,
            size_hint=(0.9, 0.9)
        )

        submit_button.bind(
            on_release=lambda x: self.info_submitted(text_input.text, popup)
        )

        content.add_widget(text_input)
        content.add_widget(submit_button)

        popup.open()

    @staticmethod
    def file_selected(path, popup):
        logger.info(f"File selected: {path}")
        popup.dismiss()

    @staticmethod
    def info_submitted(info, popup):
        logger.info(f"Information submitted: {info}")
        popup.dismiss()
