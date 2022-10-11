from cv2 import transform
import torch
import torchvision
from torch.utils.data import Dataset

# If you use Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NameDataset(torchvision.datasets.CocoDetection):
    def __init__(self, transform = None,*kwargs):
        super().__init__()
        self.transform = transform
        self.kwargs = kwargs

    def __getitem__(self, idx):
        
        return {'image':...,
                'label':...}

# Add you other methods under