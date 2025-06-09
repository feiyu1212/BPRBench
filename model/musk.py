import torch
import torchvision
from transformers import XLMRobertaTokenizer

from musk import utils, modeling
from timm.models import create_model
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from model.base_clip import BaseClip  # Make sure this module is available in your PYTHONPATH


class MuskModel:
    def __init__(self, ckpt=''):
        self.model_name = 'MUSK'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self.model, 'model|module', '')
        self.model = self.model.to(self.device).eval()
        self.model_tokenizer = XLMRobertaTokenizer("projects/MUSK/musk/models/tokenizer.spm")

    def encode_text(self, text):
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
        text_ids = []
        paddings = []
        for txt in texts:
            txt_ids, pad = utils.xlm_tokenizer(txt, self.model_tokenizer, max_len=100)
            text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
            paddings.append(torch.tensor(pad).unsqueeze(0))

        text_ids = torch.cat(text_ids)
        paddings = torch.cat(paddings)
        with torch.inference_mode():
            text_embeddings = self.model(
                text_description=text_ids.to(self.device),
                padding_mask=paddings.to(self.device),
                with_head=True,
                out_norm=True
            )[1]  # return (vision_cls, text_cls)
        return text_embeddings

    def encode_image(self, image):
        with torch.inference_mode():
            image_embeddings = self.model(
                image=image.to(self.device),
                with_head=True,  # We only use the retrieval head for image-text retrieval tasks.
                out_norm=True
            )[0]  # return (vision_cls, text_cls)
        return image_embeddings

class Musk(BaseClip):
    def __init__(self, ckpt=''):
        super().__init__(ckpt)
        self.model_name = 'MUSK'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _init_model(self, ckpt):
        self.tokenizer = None
        self.model = MuskModel()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(384, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        
