import sys

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

class DeepSeekJanusPro:
    def __init__(self, model_path: str='LLM/Janus-Pro-7B'):
        # Initialize the model and tokenizer
        self.model_path = model_path
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def generate(self, system_prompt: str, user_prompt: str, image_paths: list = []):
        # Prepare conversation input
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{user_prompt}",
                "images": image_paths,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Load images if any and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        # Run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        # Decode the response and return it
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # return f"{prepare_inputs['sft_format'][0]}", answer
        return answer
