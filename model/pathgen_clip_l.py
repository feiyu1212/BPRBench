import torch
import open_clip
from model.base_clip import BaseClip  # Make sure this module is available in your PYTHONPATH

class PathGenClipL(BaseClip):
    def __init__(self, ckpt):
        super().__init__(ckpt)
        self.model_name = 'pathgen-clip-l'
    
    def _init_model(self, ckpt):
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.transform = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained=ckpt)
        self.model = self.model.to(self.device).eval()

