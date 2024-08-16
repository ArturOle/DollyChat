import time
import random


class FakeDolly:
    """ Fake Dolly model for testing purposes. """
    excuses = [
        "I'm on a coffee break.",
        "I'm on a lunch break.",
        "I'm doing my laundry.",
        "I'm in a meeting.",
        "I'm doing my dishes.",
        "I'm cleaning my workspace.",
        "I'm walking my dog.",
        "I'm walking my cat.",
        "I'm walking my fish.",
        "I'm preparing my dinner.",
        "I'm preparing my breakfast.",
        "I'm preparing my coffee.",
        "SELECT excuse FROM SemiBuissnessRelatedExcusses ORDER BY RANDOM() LIMIT 1;"
    ]

    def __init__(self):
        self.model = None
        print("I'm bering innitiated!")

    def generate_response(self, prompt: str):
        """ Generates a fake response. """
        time.sleep(5)
        return f"""
            Sorry, I'm working remotely and {random.choice(self.excuses)}
            Please text me later.
        """
