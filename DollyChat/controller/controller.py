import asyncio
import logging


class Controller:
    def __init__(self, model, view):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        self.model = model
        self.view = view
        self.response = None

    def update_data(self, data):
        self.model.set_data(data)
        self.view.data_text = self.model.get_data()
        self.view.text_input.text = ""

    def generate_response(self, data):
        self.tmp_data_text = "Generating response..."
        self.response = self.model.generate_response(data)
        self.logger.info(f"Response generated in Controller:\n {self.response}")
        self.view.data_text = self.tmp_data_text
        self.view.text_input.text = ""

    async def fake_task(self):
        self.logger.info("Fake task started")
        asyncio.sleep(5)
        self.response = "Fake response"
        self.logger.info("Fake task ended")
        return "Fake task ended"

    # async def wait_for_response(self, original):
    #     dot_count = 0
    #     while "Generating response" in self.tmp_data_text:
    #         dot_count += 1
    #         self.view.data_text = self.view.data_text.replace(
    #             "Generating response"+'' * dot_count % 3, original
    #         )
    #         self.view.update_data(self.view.data_text)
    #         await asyncio.sleep(0.1)
