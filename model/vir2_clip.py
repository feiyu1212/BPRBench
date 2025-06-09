import sys
import torch

sys.path.append('CONCH/')
sys.path.append('hipa_clip')
from vir2_clip import get_vir2_clip_tokenizer, get_vir2_clip
from model.base_clip import BaseClip


class Vir2Clip(BaseClip):
    def __init__(self, ckpt):
        super().__init__(ckpt)
        self.model_name = 'vir2-clip'

    def _init_model(self, ckpt):
        self.tokenizer = get_vir2_clip_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.transform = get_vir2_clip(checkpoint=ckpt, remove_text=False, remove_visual=False)
        self.model = self.model.to(self.device).eval()