import base64
from openai import OpenAI

class InternVL2_5_78B:
    def __init__(self, model_name, api_key, base_url):
        """Initialize the InternVL2_5_78B with API key and base URL."""
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    
    def _image_to_base64(self, image_path):
        """Convert image to Base64 encoded string."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def generate(self, system, prompt, image_paths):
        """
        Generate response based on a user-provided prompt and image paths.
        
        :param system: The system message.
        :param prompt: The text prompt to accompany the image.
        :param image_paths: List of image file paths.
        
        :return: The model's response.
        """
        if not image_paths:
            raise ValueError("No image paths provided.")
        
        # Prepare messages for the conversation
        messages = [
            {"role": "system", "content": system}
        ]
        
        # Prepare content list for user message
        user_content = []
        
        # Process each image path
        for image_path in image_paths:
            base64_image = self._image_to_base64(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image;base64,{base64_image}"
                }
            })
        
        # Add text prompt
        user_content.append({"type": "text", "text": prompt})
        
        # Add user message with content
        messages.append({"role": "user", "content": user_content})

        # Call the API for the response
        response = self.client.chat.completions.create(
            model="OpenGVLab/InternVL2_5-78B-MPO",
            messages=messages,
            temperature=0,
            stream=False,
        )
        
        # Extract and return the response
        result_text = response.choices[0].message.content
        # print(result_text)
        return result_text

