import torch
import torch.nn as nn
from models.resnet_attn import CrossAttention
import torch.nn.functional as F


class DualAttentionModel(nn.Module):
    def __init__(self, model1, model2, feature_dim=512, fc_out_dim=256):
        super(DualAttentionModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # self.cross = cross
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.cross_attention = CrossAttention(embed_dim=2048)
        # self.softmax_weights = nn.Parameter(torch.randn(2))
        # self.fc_cross =nn.Sequential(nn.Linear(4096, 2048), nn.Linear(2048, 1024))
        self.fc = nn.Linear(feature_dim*2, fc_out_dim)
        self.fc_add = nn.Linear(feature_dim , fc_out_dim)

    def forward(self, input1, input2, return_embeddings=False):
        features1 = self.model1(input1)  
        features2 = self.model2(input2)  
        
        # if self.cross:
        #     cross_attn_output1 = self.cross_attention(features1, features2)
        #     cross_attn_output2 = self.cross_attention(features2, features1)
            
        #     stacked1 = torch.stack([cross_attn_output1], dim=-1)         
        #     weights1 = F.softmax(self.softmax_weights[0], dim=0).view(1, 1, 1, 1, -1)
        #     weighted_output1 = torch.sum(stacked1 * weights1, dim=-1)
        #     weighted_output1 = self.avgpool(weighted_output1)
        #     weighted_output1 = torch.flatten(weighted_output1, 1)
            
        #     stacked2 = torch.stack([cross_attn_output2], dim=-1)         
        #     weights2 = F.softmax(self.softmax_weights[1], dim=0).view(1, 1, 1, 1, -1)  
        #     weighted_output2 = torch.sum(stacked2 * weights2, dim=-1)
        #     weighted_output2 = self.avgpool(weighted_output2)  
        #     weighted_output2 = torch.flatten(weighted_output2, 1)
        
        #     concatenated = torch.cat([weighted_output1, weighted_output2], dim=1)  
        #     # print("concatenate" ,concatenated.shape)
        #     combined = self.fc_cross(concatenated)
        #     # print("combined", combined.shape)
        
        # else:
        # combined = torch.cat([features1, features2], dim=1)  
        combined = features1 * features2
        # print(combined.shape)
        
        out = self.fc_add(combined)
        # out = self.fc(combined)

        if return_embeddings:
            return out, combined  
        return out


class SingleAttentionModel(nn.Module):
    def __init__(self, model, feature_dim=512, fc_out_dim=256):
        super(SingleAttentionModel, self).__init__()
        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, fc_out_dim)  # No need to double feature_dim

    def forward(self, input, return_embeddings=False):
        features = self.model(input)      
        # If the model outputs feature maps, apply avg pooling
        # features = self.avgpool(features)
        features = torch.flatten(features, 1)
    
        out = self.fc(features)

        if return_embeddings:
            return out, features  
        return out
