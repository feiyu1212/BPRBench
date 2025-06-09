import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

class LLaVAInternLM2:
    def __init__(self):
        self.model_name = 'LLaVA-internLM2-7B'
        base_path = '.cache/huggingface/hub'
        self.model_path = os.path.join(base_path, 'xtuner/llava-internlm2-7b')
        
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path,
            None,
            self.model_path,
            load_8bit=False,
            load_4bit=False,
            device="cuda"
        )
        
    def generate(self, system_prompt: str='', user_prompt: str='', image_paths: list = []):
        if not image_paths:
            return "No image provided"
            
        # Process images
        images = [Image.open(img_path) for img_path in image_paths]
        images = [process_images(image, self.image_processor) for image in images]
        
        # Create conversation
        conv = conv_templates["llava_v1"].copy()
        if system_prompt:
            conv.system = system_prompt
        
        # Add user prompt with image
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        
        # Generate response
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX=self.tokenizer.encode("<image>")[1], return_tensors='pt').unsqueeze(0).cuda()
        
        # Generate with images
        output_ids = self.model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True
        )
        
        # Decode output
        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return output

if __name__ == "__main__":
    model = LLaVAInternLM2()
    output = model.generate("", "describe the image", ["../data/test.jpg"])
    print("Test ", model.model_name, output) 