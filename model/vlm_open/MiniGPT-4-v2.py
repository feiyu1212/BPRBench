from typing import List
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)
from PIL import Image
import os
import torch

class MiniGPT4v2:
    def __init__(self):
        self.model_name = 'MiniGPT-4-v2'
        base_path = '.cache/huggingface/hub'
        self.model_path = os.path.join(base_path, 'alv001/MiniGpt-4-7B')
        
        # Load image processor, tokenizer, and model
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )

    def generate(self, system_prompt: str = '', user_prompt: str = '', image_paths: List[str] = None) -> str:
        """
        Generates text based on the provided user prompt and image(s).

        Parameters:
            system_prompt (str): System-level prompt (optional)
            user_prompt (str): User prompt for generation
            image_paths (List[str]): List of image paths to describe

        Returns:
            str: Generated text response
        """
        if not image_paths:
            raise ValueError("Image paths must not be empty")

        # Load and process images
        images = [Image.open(path).convert("RGB") for path in image_paths]
        image_inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)

        # Combine system and user prompt
        full_prompt = system_prompt + user_prompt if system_prompt else user_prompt
        text_inputs = self.tokenizer([full_prompt], padding="max_length", return_tensors="pt").to(self.model.device)

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                vision_inputs=image_inputs.pixel_values,
                text_input_ids=text_inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.95
            )

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.strip()

if __name__ == "__main__":
    model = MiniGPT4v2()
    print(model.generate("", "describe the image", ["../data/test.jpg"]))