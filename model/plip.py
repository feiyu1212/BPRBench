import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
import sys
import torch

sys.path.append('projects/plip')
from plip import PLIP as PLIPBase
from model.base_clip import BaseClip
from PIL import Image
import numpy as np
import clip
from torchvision import transforms

class PLIPModel:
    def __init__(self, ckpt=''):
        self.model = PLIPBase('vinid/plip')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model #.to(self.device).eval()

    def encode_text(self, text):
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
        text_embeddings = self.model.encode_text(texts, batch_size=512)
        text_embeddings = text_embeddings/np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
        return torch.from_numpy(text_embeddings).to(self.device)

    def encode_image(self, image):
        image = [Image.fromarray(im.cpu().numpy().astype(np.uint8)) for im in image]
        image_embeddings = self.model.encode_images(image, batch_size=512)
        image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        return torch.from_numpy(image_embeddings).to(self.device)



class PLIP(BaseClip):
    def __init__(self, ckpt=''):
        super().__init__(ckpt)
        self.model_name = 'plip'

    def _init_model(self, ckpt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PLIPModel()
        self.tokenizer = None
        # _, self.transform = clip.load("ViT-B/32", device=self.device)
    
    def transform(self, image):
        return np.array(image.resize((224, 224)))

if __name__ == "__main__":
    from PIL import Image
    model = PLIP()
    text = "histopathology image of lung adenocarcinoma"
    # text_embeddings = model.model.encode_text(text)
    # print(text_embeddings.shape)

    image = Image.open("chat_data/breast_test/NORM/D23-21097-3-4_68096_67648_image.jpg")
    # print(type(model.transform(image)))
    image_embeddings = model.model.encode_image(model.transform([image]))
    print(image_embeddings.shape)

