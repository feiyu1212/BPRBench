import requests
import base64
import json
import os

from openai import AzureOpenAI


class GPT4oVisionModel:
    def __init__(self, api_key=None, endpoint=None):
        # Use provided credentials or fall back to environment variables
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-02-01"
        )

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
            model="gpt-4o-global",  # model = "deployment_name"
            messages=[
                { "role": "system", "content": system},
                { "role": "user", "content": images_content + [{ "type": "text", "text": prompt}] }
            ]
        )
        return response.choices[0].message.content
