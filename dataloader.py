import pytorch_lightning as pl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2

    
class NameDataModule(pl.LightningDataModule):
    def __init__(self,batch_size=2):
        """
        No test annotation ?
        
        
        """
        super().__init__()
        self.dataset_dir = os.environ.get('DSDIR')+'/COCO/'        
        self.batch_size = batch_size

    def prepare(self):
        build_dataset(image_set='train', path = self.dataset_dir)
        build_dataset(image_set='val',path = self.dataset_dir) 
        
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.COCOtrain =  build_dataset(image_set='train', path = self.dataset_dir)
            self.COCOvalid =  build_dataset(image_set='val',path = self.dataset_dir) 
        if stage == "test" or stage is None:
            # TODO : Change after
            self.COCOtest = build_dataset(image_set='val',path = self.dataset_dir) 
            
    def train_dataloader(self):
        return DataLoader(self.COCOtrain, batch_size = self.batch_size, drop_last = False,shuffle = True, collate_fn= collate_fn)
    def val_dataloader(self):
        return DataLoader(self.COCOvalid, batch_size = self.batch_size, drop_last =  False, collate_fn= collate_fn)        
    def test_dataloader(self):
        return DataLoader(self.COCOtest, batch_size = self.batch_size, drop_last =  False, collate_fn= collate_fn)