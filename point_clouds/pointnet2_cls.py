'''
source code and pretrained model: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master
'''
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(PointNet2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

import torch

class PointNet2FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model, normal_channel =True):
        super(PointNet2FeatureExtractor, self).__init__()
        self.normal_channel = normal_channel
        in_channel = 6 if normal_channel else 3
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])        
        self.sa1 = pretrained_model.sa1
        self.sa2 = pretrained_model.sa2
        self.sa3 = pretrained_model.sa3

    def forward(self, xyz):
        if xyz.shape[1] == 3:  # If only XYZ coordinates are present
            zeros = torch.zeros_like(xyz)  # Create fake normal vectors
            xyz = torch.cat([xyz, zeros], dim=1)  # Concatenate to get shape [B, 6, N]
        # print(xyz.shape)
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 3,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            # conv1 = self.sa1.conv_blocks[0][0]
            # self.sa1.conv_blocks[0][0] = nn.Conv2d(3, conv1.out_channels, kernel_size=1, bias=False)

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.shape)
        # print(l3_points.shape)
        # x = l3_points.view(B, 1024)
        # x = torch.max(x, dim=-1)[0]  # Max pooling to reduce to a fixed-size feature vector
        # print(x.shape)
        return _,l3_points