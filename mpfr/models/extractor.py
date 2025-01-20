import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

class TimmFeatureExtractor(nn.Module):
    
    def __init__(self, 
                 model_name="vit_small_patch14_dinov2", 
                 weight_path='/home/c1c/workbench/ckpts/dinov2_vits14_pretrain.pth', 
                 pretrained=True,
                 image_size=256,
                 feature_size=64, 
                 stages=[11]):
        super(TimmFeatureExtractor, self).__init__()
        
        
        self.model = timm.create_model(
            model_name=model_name,
            img_size=image_size,
            pretrained=pretrained,
            features_only=True,
            pretrained_cfg_overlay=dict(file=weight_path),
            out_indices=stages,
        )
        
        self.image_size = image_size
        self.feature_size = feature_size
        self.stages = stages

        for param in self.model.parameters():
            param.requires_grad = False

    def count(self):
        fake_batch = torch.randn(1, 3, self.image_size, self.image_size).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            fake_features = self.model(fake_batch)
            
        channels = []
        for fake_feat in fake_features:
            if fake_feat.shape[2] != fake_feat.shape[3]:
                channels.append(fake_feat.shape[3])
            else:
                channels.append(fake_feat.shape[1])
        
        return channels

    def forward(self, x):
        features = self.model(x) # tuple 
        if self.feature_size:
            resized_features = []
            for feat in features:
                if feat.shape[2] != feat.shape[3]:
                    feat = feat.permute(0, 3, 1, 2).contiguous()          
                resized_feat = F.interpolate(feat, size=self.feature_size, mode='bilinear', align_corners=True)
                resized_features.append(resized_feat)
            resized_features = torch.cat(resized_features, dim=1)
        else:
            resized_features = torch.cat(features, dim=1)
        
        return resized_features

if __name__ == '__main__':
    
    # efficientnet_b4.ra2_in1k
    # tf_efficientnet_b4.aa_in1k
    # tf_efficientnet_b4.in1k

     model = timm.create_model(
            model_name='tf_efficientnet_b4.aa_in1k',
            # img_size=224,
            pretrained=True,
            features_only=True,
            pretrained_cfg_overlay=dict(file='/home/c1c/workbench/ckpts/tf_efficientnet_b4_aa-818f208c.pth'),
            out_indices=[0,1,2,3,4],
        )
     
     x = torch.randn(4,3,224,224)
     y = model(x)
     for i in y:
         print(i.shape)
    
    # import timm
    # model = timm.create_model('vit_small_patch14_dinov2', pretrained=True, img_size=512, features_only=True, out_indices=(-3, -2,))
    # output = model(torch.randn(2, 3, 512, 512))
    # for o in output:    
    #     print(o.shape) 
    
    # image_size = 512
    # inputs = torch.randn(4, 3, image_size, image_size).to('cuda:0')
    # net = TimmFeatureExtractor(
    #     model_name='vit_small_patch14_dinov2',
    #     weight_path='/home/gysj_cyc/workbench/ckpts/dinov2_vits14_pretrain.pth',
    #     # model_name='swin_large_patch4_window12_384',
    #     # weight_path='/home/gysj_cyc/workbench/ckpts/swin_large_patch4_window12_384_22kto1k.pth',
    #     pretrained=True,
    #     image_size=image_size,
    #     feature_size=64,
    #     stages=[11]).to('cuda:0')
    # print(net.count())
    # out = net(inputs)
    # print(out.shape)
