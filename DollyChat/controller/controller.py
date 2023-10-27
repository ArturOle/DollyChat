

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def update_data(self, data):
        self.model.set_data(data)
        self.view.data_text = self.model.get_data()
        self.view.text_input.text = ""
