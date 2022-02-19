import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from pointnet2.pointnet2_modules import PointnetSAModuleVotes

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True, in_channels=3):
        super(ModifiedResnet, self).__init__()

        #self.model = psp_models['resnet18'.lower()](in_channels=in_channels)
        self.model = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', in_channels=in_channels)
        #self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        #self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.bn2 = torch.nn.BatchNorm1d(128)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        #self.e_bn1 = torch.nn.BatchNorm1d(64)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.e_bn2 = torch.nn.BatchNorm1d(128)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        #self.bn5 = torch.nn.BatchNorm1d(512)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        #self.bn6 = torch.nn.BatchNorm1d(1024)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):

        #"pointnet" layer
        #XYZ -> 64 dim embedding
        x = F.relu(self.conv1(x))

        #32 dim embedding -> 64 dim embedding
        emb = F.relu(self.e_conv1(emb))

        #concating them
        pointfeat_1 = torch.cat((x, emb), dim=1)

        #64 dim embedding -> 128 dim embedding
        x = F.relu(self.conv2(x))
        #64 dim embedding -> 128 dim embedding
        emb = F.relu(self.e_conv2(emb))

        #concating them
        pointfeat_2 = torch.cat((x, emb), dim=1)

        #lifting fused 128 + 128 -> 512
        x = F.relu(self.conv5(pointfeat_2))

        #lifting fused 256 + 256 -> 1024
        x = F.relu(self.conv6(x))

        #average pooling on into 1 1024 global feature
        ap_x = self.ap1(x)

        #repeat it so they can staple it onto the back of every pixel/point
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        #64 + 64 (level 1), 128 + 128 (level 2), 1024 global feature
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, num_rot_bins, num_image_channels=3):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet(in_channels=num_image_channels)
        self.feat = PoseNetFeat(num_points)
        
        self.conv1_front = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_rot_bins = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_front = torch.nn.Conv1d(640, 256, 1)
        self.conv2_rot_bins = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)

        self.conv3_front = torch.nn.Conv1d(256, 128, 1)
        self.conv3_rot_bins = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)

        self.conv4_front = torch.nn.Conv1d(128, num_obj*3, 1) #front axis
        self.conv4_rot_bins = torch.nn.Conv1d(128, num_obj*num_rot_bins, 1) #rotation bins around front axis
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation

        self.num_obj = num_obj
        self.num_rot_bins = num_rot_bins

    def forward(self, img, x, choose, obj):

        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        
        orig_x = x
        x = x.transpose(2, 1).contiguous()

        #x is pointcloud
        #emb is cnn embedding
        ap_x = self.feat(x, emb)

        fx = F.relu(self.conv1_front(ap_x))
        rx = F.relu(self.conv1_rot_bins(ap_x))
        tx = F.relu(self.conv1_t(ap_x))     

        fx = F.relu(self.conv2_front(fx))
        rx = F.relu(self.conv2_rot_bins(rx))
        tx = F.relu(self.conv2_t(tx))

        fx = F.relu(self.conv3_front(fx))
        rx = F.relu(self.conv3_rot_bins(rx))
        tx = F.relu(self.conv3_t(tx))

        fx = self.conv4_front(fx).view(bs, self.num_obj, 3, self.num_points)
        rx = self.conv4_rot_bins(rx).view(bs, self.num_obj, self.num_rot_bins, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)

        obj = obj.unsqueeze(-1).unsqueeze(-1)
        obj_fx = obj.repeat(1, 1, fx.shape[2], fx.shape[3])
        obj_rx = obj.repeat(1, 1, rx.shape[2], rx.shape[3])
        obj_tx = obj.repeat(1, 1, tx.shape[2], tx.shape[3])

        out_fx = torch.gather(fx, 1, obj_fx)[:,0,:,:]
        out_rx = torch.gather(rx, 1, obj_rx)[:,0,:,:]
        out_tx = torch.gather(tx, 1, obj_tx)[:,0,:,:]
    
        out_fx = out_fx.contiguous().transpose(2, 1).contiguous()
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_fx + orig_x, out_rx, out_tx + orig_x, emb.detach()
