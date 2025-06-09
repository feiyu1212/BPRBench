import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

class YiVL6B:
    def __init__(self):
        # Load and initialize the model
        self.model_name = 'Yi-VL-6B'
        base_path = '.cache/huggingface/hub'
        self.model_path = os.path.join(base_path, '01ai/Yi-VL-6B')
        
        # Disable torch default weight initialization if necessary.
        disable_torch_init()
        
        # Load the tokenizer, model, image processor, and context length.
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path,
            None,
            self.model_path,
            load_8bit=False,
            load_4bit=False,
            device="cuda"
        )

    def generate(self, system_prompt: str = '', user_prompt: str = '', image_paths: list = []):
        # If no image is provided, return an appropriate message.
        if not image_paths:
            return "No image provided"
        
        # Open and process each image.
        images = [Image.open(img_path) for img_path in image_paths]
        images = [process_images(img, self.image_processor) for img in images]
        
        # Create a conversation using an appropriate template.
        # If a dedicated 'yi_vl' template exists, use that; otherwise, fall back to 'llava_v1'.
        template_key = "yi_vl" if "yi_vl" in conv_templates else "llava_v1"
        conv = conv_templates[template_key].copy()
        
        if system_prompt:
            conv.system = system_prompt
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        
        # Construct the prompt from the conversation.
        prompt = conv.get_prompt()
        
        # Obtain the image token index from the tokenizer.
        image_token_encoding = self.tokenizer.encode("<image>")
        image_token_index = image_token_encoding[1] if len(image_token_encoding) > 1 else None
        
        # Tokenize the prompt and embed image tokens.
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX=image_token_index,
            return_tensors='pt'
        ).unsqueeze(0).to("cuda")
        
        # Generate responses from the model using the processed images.
        output_ids = self.model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True
        )
        
        # Decode the generated tokens to produce the final answer.
        generated_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        return generated_text

if __name__ == "__main__":
    model = YiVL6B()
    output = model.generate("", "describe the image", ["../data/test.jpg"])
    print("Test ", model.model_name, output) 