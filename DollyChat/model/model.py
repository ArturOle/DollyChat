
class Model:
    def __init__(self):
        self.data = ""

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data


class LocalDolly:
    def __init__(self):
        self.model = None


class RemoteDolly:
    def __init__(self):
        self.model = None