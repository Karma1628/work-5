import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/mnt/SSD8T/home/cyc/workbench/paper5_memory_onenip')

from models.extractor import TimmFeatureExtractor
from models.model_onenip import OneNIP

class FeatureReconstruction(nn.Module):

    def __init__(self, 
                 model_name='vit_small_patch14_dinov2',
                 weight_path='/mnt/SSD8T/home/cyc/workbench/ckpts/dinov2_vits14_pretrain.pth',
                 image_size=224,
                 feature_size=16,
                 stages=[11],
                 num_encoder=4,
                 num_decoder=4
                 ):
        super(FeatureReconstruction, self).__init__()
        
        self.image_size = image_size

        self.feature_extractor = TimmFeatureExtractor(
            model_name=model_name,
            weight_path=weight_path,
            pretrained=True,
            image_size=image_size,
            feature_size=feature_size,
            stages=stages
        )
        
        self.stage_channels = self.feature_extractor.count() # 192, 384, 768
        ch_in = sum(self.stage_channels)
        
        self.reconstructive_network = OneNIP(
            inplanes=ch_in,
            feature_size=[feature_size, feature_size],
            num_decoder_layers=num_decoder,
            num_encoder_layers=num_encoder,
            image_size=image_size
        )
        
    def forward(self, image, prompt, pseudo_image=None):

        #*------------------------------------ Feature Extraction ------------------------------------*#
        with torch.no_grad():
            if self.training:
                images = torch.cat([image, prompt, pseudo_image], dim=0)
                batchsize = image.shape[0]
                features = self.feature_extractor(images)
                feature = features[:batchsize, ...]
                prompt_feature = features[batchsize:2*batchsize, ...]
                pseudo_feature = features[2*batchsize:3*batchsize, ...]
            else:
                images = torch.cat([image, prompt], dim=0)
                batchsize = image.shape[0]
                features = self.feature_extractor(images)
                feature = features[:batchsize, ...]
                prompt_feature = features[batchsize:, ...]
                pseudo_feature = feature

        #*---------------------------------- Feature Reconstruction & Segmentation----------------------------------*#
        inputs = {}
        inputs["pseudo_feature"] = pseudo_feature
        inputs["prompt_feature"] = prompt_feature
        outputs = self.reconstructive_network(inputs)

        return {
            'feature': feature,
            'reconstruction': outputs["rec_feat"],
            'prediction': outputs["ref_pred"],
            'latent': outputs["latent"]
        }

if __name__ == "__main__":
    size = 224
    x = torch.randn(4, 3, size, size).to('cuda:0')
    y = torch.randn(4, 3, size, size).to('cuda:0')
    z = torch.randn(4, 3, size, size).to('cuda:0')
    net = FeatureReconstruction(
        stages=[11],
        image_size=size,
        feature_size=16
    ).to('cuda:0')
    net.eval()
    out = net(x, y, None)
    for key in out.keys():
        print(key, out[key].shape)