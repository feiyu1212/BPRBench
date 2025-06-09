import base64
import random

class RandomModel:
    def __init__(self):
        self.model_name = 'Random'

    def generate(self, system, prompt, image_paths):
        valid_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        options = []
        for line in prompt.split('\n'):
            option = line.split(' ')[0]
            if option in valid_options:
                options.append(option)

        result = random.choice(options)
        return result
