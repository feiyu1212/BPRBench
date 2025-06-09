import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

class CogVLM:
    def __init__(self):
        # Model name and model cache path (adjust the base_path as needed)
        self.model_name = 'CogVLM'
        
        # Load the tokenizer (make sure the vicuna tokenizer is installed)
        self.tokenizer = LlamaTokenizer.from_pretrained('ZhipuAI/CogVLM')
        
        # Set device and torch dtype (using fp16 by default)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16  # or torch.bfloat16 if bf16 is preferred
        
        # Load the model.
        # In the demo code, the default pretrained checkpoint is THUDM/cogagent-chat-hf.
        # Adjust parameters (e.g., 4-bit quantization) as needed.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, system_prompt: str = '', user_prompt: str = '', image_paths: list = []):
        """
        Generate an answer given a user prompt and optional image paths using the visual language model.

        Parameters:
            system_prompt (str): Optional system instructions (not used in this simple demo).
            user_prompt (str): The main text prompt from the user.
            image_paths (list): A list of image paths; only the first image (if provided) is used.

        Returns:
            str: The generated response from the model.
        """
        # Load image if an image path is provided (only one image is supported per conversation)
        image = None
        if image_paths:
            try:
                image = Image.open(image_paths[0]).convert("RGB")
            except Exception as e:
                print(f"Error loading image from {image_paths[0]}: {e}")
                return ""
        
        # History of the conversation is empty for a single-turn inference.
        history = []
        query = user_prompt  # system_prompt is ignored in this demo
        
        # Build model input using the custom method 'build_conversation_input_ids' if available.
        if image is None:
            # For text-only conversation, we set template_version to 'base'
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer, query=query, history=history, template_version='base'
            )
        else:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer, query=query, history=history, images=[image]
            )
        
        # Prepare the inputs for the model. We need to unsqueeze to add batch dimension and move tensors to the correct device.
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(self.device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(self.device),
        }
        if image is not None and "images" in input_by_model:
            # Wrap the image tensor in a double list as expected by the model
            inputs["images"] = [[input_by_model["images"][0].to(self.device).to(self.torch_dtype)]]
        
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [[input_by_model["cross_images"][0].to(self.device).to(self.torch_dtype)]]
        
        # Define generation parameters; adjust max_length, temperature etc. as needed.
        gen_kwargs = {"max_length": 2048, "do_sample": False}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            # Remove the conversation prompt from output
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Optionally, trim the response at the termination token "</s>"
            if "</s>" in response:
                response = response.split("</s>")[0]
        
        return response

if __name__ == "__main__":
    # Basic test to check the implementation
    model = CogVLM()
    print(model.generate("", "describe the image", ["../data/test.jpg"]))