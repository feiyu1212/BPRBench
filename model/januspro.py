import requests
import base64
import json
import os
from openai import OpenAI


class DeepSeekJanusPro:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = os.environ.get("DEEPSEEK_ENDPOINT")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def _image_to_base64(self, image_path):
        """Convert image to Base64 encoded string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def generate(self, system_prompt, user_prompt, image_paths=[]):
        """Generate response with images and text prompts"""
        messages = [{"role": "system", "content": system_prompt}]
        
        content = []
        # Handle images
        for image_path in image_paths:
            base64_image = self._image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            })
        
        # Add text prompt
        content.append({"type": "text", "text": user_prompt})
        
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model="janus-pro",
            messages=messages,
            temperature=0.5
        )
        
        return response.choices[0].message.content

