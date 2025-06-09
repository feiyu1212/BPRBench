import sys
import torch

sys.path.append('CONCH/')
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from model.base_clip import BaseClip


class ConchTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        return tokenize(texts=texts, tokenizer=self.tokenizer)


class Conch(BaseClip):
    def __init__(self, ckpt):
        super().__init__(ckpt)
        self.model_name = 'conch'

    def _init_model(self, ckpt):
        model_cfg = 'conch_ViT-B-16'
        force_image_size = 224
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.transform = create_model_from_pretrained(model_cfg, checkpoint_path=ckpt, force_image_size=force_image_size)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = ConchTokenizerWrapper(get_tokenizer())