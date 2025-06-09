import base64
import os
from openai import AzureOpenAI

class GPTo1Model:
    def __init__(self, api_key=None, endpoint=None, deployment="o1", api_version="2024-12-01-preview"):
        # Use provided credentials or fall back to environment variables
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment = deployment
        self.api_version = api_version
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

    def _image_to_base64(self, image_path):
        """Convert image to Base64 encoded string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def generate(self, system, prompt, image_paths=None, timeout=60):
        """Generate response based on system prompt, user prompt and optional images"""
        messages = [
            {"role": "system", "content": system}
        ]
        
        if image_paths:
            images_content = []
            for f in image_paths:
                base64_image = self._image_to_base64(f)
                images_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            # Add user message with images and text
            messages.append({
                "role": "user", 
                "content": images_content + [{"type": "text", "text": prompt}]
            })
        else:
            # Add user message with text only
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            timeout=timeout
        )
        
        return response.choices[0].message.content
