import argparse
import base64
import time
from io import BytesIO
from typing import Dict, List

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

try:
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
except ImportError:
    print("Error: Please install quilt-llava dependencies.")
    raise SystemExit(1)


app = FastAPI()
engine = None


def load_image(url_or_data: str) -> Image.Image:
    if url_or_data.startswith("http"):
        resp = requests.get(url_or_data, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    elif url_or_data.startswith("data:image"):
        _, b64 = url_or_data.split(",", 1)
        img = Image.open(BytesIO(base64.b64decode(b64)))
    else:
        img = Image.open(BytesIO(base64.b64decode(url_or_data)))
    return img.convert("RGB")


class SimpleEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        disable_torch_init()
        self.model_name = get_model_name_from_path(args.model_path)
        print(f"Loading model: {self.model_name}...")

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path,
            args.model_base,
            self.model_name,
            args.load_8bit,
            args.load_4bit,
            device=args.device,
        )

        self.conv_mode = "llava_v1"
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        if "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        if "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"

        self.image_aspect_ratio = args.image_aspect_ratio

    def generate(self, messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.2):
        image_urls = []
        conv = conv_templates[self.conv_mode].copy()

        if not messages:
            raise ValueError("messages must not be empty")

        if messages[0]["role"] == "system":
            conv.system = messages[0]["content"]
            messages = messages[1:]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_part = ""

            if isinstance(content, str):
                text_part = content
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        text_part += item["text"]
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"] if isinstance(item["image_url"], dict) else item["image_url"]
                        image_urls.append(url)

            if role == "user":
                if DEFAULT_IMAGE_TOKEN not in text_part and len(image_urls) > 0:
                    if conv.get_prompt().find(DEFAULT_IMAGE_TOKEN) == -1:
                        prefix = (DEFAULT_IMAGE_TOKEN + "\n") * len(image_urls)
                        text_part = prefix + text_part
                conv.append_message(conv.roles[0], text_part)
            elif role == "assistant":
                conv.append_message(conv.roles[1], text_part)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = None
        if image_urls:
            images = [load_image(url) for url in image_urls]
            image_processor_cfg = type("Args", (), {"image_aspect_ratio": self.image_aspect_ratio})()
            images_tensor = process_images(images, self.image_processor, image_processor_cfg)
            if isinstance(images_tensor, list):
                images_tensor = [x.to(self.model.device, dtype=torch.float16) for x in images_tensor]
            else:
                images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=(temperature > 0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]

        return outputs.strip(), output_ids.shape[1]


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.2)
    max_tokens = data.get("max_tokens", 512)

    try:
        output_text, total_tokens = engine.generate(messages, max_tokens, temperature)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": total_tokens,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")

    args = parser.parse_args()
    engine = SimpleEngine(args)
    uvicorn.run(app, host=args.host, port=args.port)
