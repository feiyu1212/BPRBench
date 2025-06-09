import base64
import json
from dashscope import MultiModalConversation
import dashscope

dashscope.api_key = ''

class QwenVisionModel:
    def __init__(self, api_key=None):
        """Initialize the QwenVisionModel with API key."""
        self.api_key = api_key
    
    def _image_to_base64(self, image_path):
        """Convert image to Base64 encoded string."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def generate(self, system, prompt, image_paths):
        """
        Generate response based on a user-provided prompt and image paths.
        Only supports handling one image path per call for now.
        
        :param prompt: The text prompt to accompany the image.
        :param image_paths: List of image file paths.
        
        :return: The model's response (either 'yes' or 'no').
        """
        if not image_paths:
            raise ValueError("No image paths provided.")
        
        # Prepare messages for the conversation
        messages = [{
            'role': 'system',
            'content': [{'text': 'You are a professional pathology expert.'}]
        }]
        messages= []
        
        # Process each image path (only supports one image for now)
        for image_path in image_paths:
            base64_image = self._image_to_base64(image_path)
            messages.append({
                'role': 'user',
                'content': [
                    {'image': f"data:image/png;base64,{base64_image}"},
                    {'text': prompt}
                ]
            })

        # Call DashScope API for the response
        response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        print(response)
        
        # Extract and return the response
        result_text = response.output.choices[0].message.content[0]['text']
        print(result_text)
        return result_text
