import torch
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class FilteredImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, allowed_classes=None):
        # Initialize the parent ImageFolder
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # Filter indices based on allowed classes
        if allowed_classes is not None:
            self.allowed_classes = allowed_classes
            self.allowed_indices = [
                idx for idx, (_, label) in enumerate(self.samples)
                if self.classes[label] in allowed_classes
            ]
            self.imgs = [self.imgs[i] for i in self.allowed_indices]
        else:
            self.allowed_indices = range(len(self.samples))
            
    def __len__(self):
        return len(self.allowed_indices)
    
    def __getitem__(self, index):
        # Map index to the original dataset
        original_index = self.allowed_indices[index]
        return super().__getitem__(original_index)


def convert_df(dump, dataset, classnames):
    # Get file paths
    files = [x[0] for x in dataset.imgs]
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(torch.from_numpy(dump['logits']).to(torch.float32), dim=1).numpy()
    
    # Create DataFrame with all columns at once
    df = pd.DataFrame({
        'file': files,
        # 'pred': dump['preds'],
        # 'pred_class': [classnames[p] for p in dump['preds']],
        **{classname: probs[:, i] for i, classname in enumerate(classnames)}
    })
    
    # Reorder columns
    columns = ['file'] + classnames
    return df[columns]