import os
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class InstructBLIP:
    def __init__(self):
        # Load and initialize the model
        self.model_name = 'InstructBLIP-7B'
        
        # Initialize model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_path = '.cache/huggingface/hub'
        self.model = InstructBlipForConditionalGeneration.from_pretrained(os.path.join(base_path, "AI-ModelScope/instructblip-vicuna-7b"))
        self.processor = InstructBlipProcessor.from_pretrained(os.path.join(base_path, "AI-ModelScope/instructblip-vicuna-7b"))
        self.model.to(self.device)

    def _generate(self, system_prompt: str='', user_prompt: str='', image_paths: list = []):
        if not image_paths or not user_prompt:
            return "Please provide both image path and prompt"
        
        try:
            # Load and process the first image (currently handling single image)
            image = Image.open(image_paths[0]).convert("RGB")
            
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=user_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            
            # Decode and return the generated text
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(user_prompt)
            print('-----')
            print(generated_text)
            return generated_text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate(self, arrs):
        return self._generate('', arrs[-1], arrs[:-1])

if __name__ == "__main__":
    model = InstructBLIP()
    output = model._generate("", "describe the image", ["../data/test.jpg"])
    print("Test ", model.model_name, output) 