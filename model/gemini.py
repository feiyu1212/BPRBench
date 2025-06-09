import os
import base64

import openai
# from dotenv import load_dotenv

# load_dotenv()

class Gemini:

    usages = []

    def __init__(self):
        self.api_key = ''
        self.api_base = 'https://generativelanguage.googleapis.com/v1beta/openai'
        self.client = openai.Client(api_key=self.api_key, base_url=self.api_base)
        self.model_name = 'gemini-2.0-flash'

    def _image_to_base64(self, image_path):
        """Convert image to Base64 encoded string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def generate(self, system, prompt, image_paths):
        """Generate response based on system prompt, user prompt and images"""
        images_content = []
        for f in image_paths:
            base64_image = self._image_to_base64(f)
            images_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                { "role": "system", "content": system},
                { "role": "user", "content": images_content + [{ "type": "text", "text": prompt}] }
            ]
        )
        return response.choices[0].message.content

