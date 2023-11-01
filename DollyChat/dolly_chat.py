from kivy.app import App
from model import Model
from view import View
from controller import Controller


class DCApp(App):
    def build(self):
        model = Model()
        view = View(Controller(model, None))
        view.controller.view = view
        return view


if __name__ == '__main__':
    DCApp().run()
