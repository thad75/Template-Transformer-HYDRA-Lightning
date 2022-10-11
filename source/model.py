# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from loss import *
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, reduce_dict)
from typing import List

class Model(pl.LightningModule):
    """ This is a PL Model """
    def __init__(self, num_queries, backbone, transformer, matcher, 
                    num_classes, 
                    num_decoder_stack = 6,
                    num_encoder_stack = 6,
                    hidden_dim = 256, 
                    multiscale= False,
                    aux_loss = False,
                    masks = False,
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100, 
                    dn_feature_noise_scale= 0.2):   
        """ Initializes the model.
        Parameters:
           
        """
        super().__init__()               
        self.save_hyperparameters()
        self.multiscale = multiscale
        self.num_feature_levels = 3 if self.multiscale else 1          # Hard-coded multiscale parameters
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        assert self.hidden_dim == transformer.d_model
        self.losses = ['labels', 'boxes', 'cardinality']
        self.backbone = backbone
        self.backbone = self.backbone.eval()
        self.transformer = transformer
        self.matcher = matcher
        self.num_classes = num_classes
        self.masks = masks
        self.weight_dict = build_weights()
        self.bbox_embed = nn.Embedding(self.num_queries, 4)  

        dataset_dir = os.environ.get('DSDIR')+'/COCO/'        
        # For Evalutation 
        self.post_processors = {'bbox': PostProcess()}
        self.base_ds = get_coco_api_from_dataset(build_dataset(image_set='val',path = dataset_dir))
        self.iou_types = tuple(k for k in ('segm', 'bbox') if k in self.post_processors.keys())
    def forward(self, samples: NestedTensor,  targets:List=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        srcs,patch_srcs = [],[]
        masks,patch_masks = [],[]

        features, pos_embeds = self.backbone(samples)
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs =srcs,
                                                                    masks = masks,
                                                                    query_embed = self.query_embed.weight,
                                                                    noised_query_bbox = input_query_bbox,
                                                                    pos_embeds = pos_embeds,
                                                                    query_label = input_query_label,
                                                                    query_features = input_query_features,
                                                                    crop=None,
                                                                    attn_mask = attn_mask)


        hs[0]+=self.label_enc.weight[0,0]*0.0
        #print('HS', hs.shape)
        outputs_coords = []
        outputs_class = []  

        for lvl in range(hs.shape[0]):
            reference_before_sigmoid = inverse_sigmoid(reference[lvl])
            bbox_offset = self.bbox_embed[lvl](hs[lvl])
            outputs_coord = (reference_before_sigmoid + bbox_offset).sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_class.append(self.class_embed[lvl](hs[lvl]))
                                                          
        outputs_coords = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_class)        

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coords[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords)
        return ...

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4,weight_decay= 1e-4) 
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 10)   
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    def training_step(self,batch, batch_idx):
        data, label = batch
        
        out = self.forward(data)
        loss = ...
        return loss
        
        
    def validation_step(self,batch,  batch_idx):
        data, label = batch
        
        out = self.forward(data)
        loss = ...
        return loss
        
        
    def on_validation_start(self):
        ...

        
    def on_validation_end(self):
        ...

 