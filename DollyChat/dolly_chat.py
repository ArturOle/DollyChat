from kivy.app import App
from model import Model
from view import View
from controller import Controller


class DCApp(App):
    def build(self):
        model = Model()
        controller = Controller(model, None)
        view = View(controller)
        controller.view = view
        controller.load_model()
        return view


if __name__ == '__main__':
    DCApp().run()
