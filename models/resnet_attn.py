import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return attn_output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))        
        return x * avg_out.expand_as(x) 
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 3):
        super(SpatialAttention, self).__init__()
        # self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return attention * x  


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        attn_output, _ = self.multihead_attn(x1, x2, x2)
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return attn_output

class AttentionResNet(nn.Module):
    def __init__(self, attention_types):
        super(AttentionResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc without flattening


        self.attention_types = attention_types
        self.attention_layers = nn.ModuleList()

        for attn in attention_types:
            if attn == "channel":
                self.attention_layers.append(ChannelAttention(2048))
            elif attn == "spatial":
                self.attention_layers.append(SpatialAttention())
            elif attn == "self":
                self.attention_layers.append(SelfAttention(2048))
            elif attn == "cross":
                self.attention_layers.append(CrossAttention(2048))

        self.fc = nn.Linear(2048, 512)
        self.softmax_weights = nn.Parameter(torch.ones(len(attention_types), requires_grad=True))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2=None):
        features = self.resnet(x1)
        attention_outputs = []

        for i, attn_layer in enumerate(self.attention_layers):
            if self.attention_types[i] == "cross" and x2 is not None:
                attn_out = attn_layer(features, self.resnet(x2))
            else:
                # print(f"Features shape before attention: {features.shape}")
                attn_out = attn_layer(features)
            if len(self.attention_types)>1:
                attention_outputs.append(attn_out.unsqueeze(-1))
            else:
                attention_outputs.append(attn_out)
        if len(attention_outputs) == 0:
            return features

        stacked = torch.stack(attention_outputs, dim=-1)  # [B, C, H, W, num_attentions]
        
        # stacked = torch.cat(attention_outputs, dim=-1)  # [B, C, H, W, num_attentions]
        weights = F.softmax(self.softmax_weights, dim=0).view(1, 1, 1, 1, -1)
        weighted_output = torch.sum(stacked * weights, dim=-1)

        weighted_output = self.avgpool(weighted_output)  # Global Pooling
        # print("Weighted Output Shape Before Flatten:", weighted_output.shape)
        weighted_output = torch.flatten(weighted_output, 1)
        
        return self.fc(weighted_output)

        
class AttentionResNet2(nn.Module):
    def __init__(self, attention_types, reduction, kernel_size):
        super(AttentionResNet2, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc without flattening


        self.attention_types = attention_types
        self.attention_layers = nn.ModuleList()

        for attn in attention_types:
            if attn == "channel":
                self.attention_layers.append(ChannelAttention(2048, reduction=reduction))
            elif attn == "spatial":
                self.attention_layers.append(SpatialAttention(kernel_size=kernel_size))
            elif attn == "self":
                self.attention_layers.append(SelfAttention(2048))
            elif attn == "cross":
                self.attention_layers.append(CrossAttention(2048))

        self.fc = nn.Linear(2048, 512)
        self.softmax_weights = nn.Parameter(torch.ones(len(attention_types), requires_grad=True))

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x1, x2=None):
        features = self.resnet(x1)
        attention_outputs = []

        for i, attn_layer in enumerate(self.attention_layers):
            if self.attention_types[i] == "cross" and x2 is not None:
                attn_out = attn_layer(features, self.resnet(x2))
            else:
                # print(f"Features shape before attention: {features.shape}")
                attn_out = attn_layer(features)
            if len(self.attention_types)>1:
                attention_outputs.append(attn_out.unsqueeze(-1))
            else:
                attention_outputs.append(attn_out)
        
        if len(attention_outputs) == 0:
            return features
        # print(f"len attn : {len(attention_outputs)}")
        # print(f"attn0 shape : {(attention_outputs[0].shape)}")
        
        normalized_outputs = [F.layer_norm(attn_out, attn_out.shape[1:]) for attn_out in attention_outputs]
        # print(f"normalized  shape : {(normalized_outputs[0].shape)}")
        stacked = torch.stack(normalized_outputs, dim=-1)  # [B, C, H, W, num_attentions]
        # print(f"stacked  shape : {(stacked.shape)}")

        temperature = 0.1
        weights = F.softmax(self.softmax_weights / temperature, dim=0).view(1, 1, 1, 1, -1)
        # print(f"weights  shape : {(weights.shape)}")

        weighted_output = torch.sum(stacked * weights, dim=-1)

        stacked = stacked.squeeze(-2)  # Remove the extra dimension

        weighted_output = self.avgpool(weighted_output)
        # print(f"weightedop shape 3 : {(weighted_output.shape)}")
        weighted_output = torch.flatten(weighted_output, 1)
        # print(f"weightedop shape 4 : {(weighted_output.shape)}")


        return self.fc(weighted_output)

