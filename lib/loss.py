from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random

from lib.transformations import rotation_matrix_from_vectors_procedure_batch, rotation_matrix_of_axis_angle_batch
from lib.loss_helper import roll_by_gather, rot_bins_loss_coeff, front_loss_coeff, translation_loss_coeff

cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')


def loss_calculation(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, t, num_rot_bins):

    #print("shapes loss regular", pred_front.shape, pred_rot_bins.shape, pred_t.shape, pred_c.shape, front_r.shape, rot_bins.shape, front_orig.shape, t.shape, model_points.shape, points.shape)

    #pred_front -> bs * num_p * 3
    #pred_rot_bins -> bs * num_p * num_rot_bins
    #pred_t -> bs * num_p * 3
    #pred_c -> bs * num_p * 1
    #model_points -> bs * num_model_points * 3
    #points -> bs * num_p * 3

    #front_r -> bs * 3
    #rot_bins -> bs * num_rot_bins
    #front_orig -> bs * 3
    #t -> bs * 1 * 3

    #orig_rot_bins -> bs * num_rot_bins
    #used for new_rot_bins calculation
    orig_rot_bins = rot_bins

    bs, num_p, _ = pred_t.size()
    
    orig_front_r = front_r

    #front_r -> bs * num_p * 3
    front_r = front_r.view(bs, 1, 3).repeat(1, num_p, 1)

    #pred_rot_bins -> bs * num_rot_bins * num_p
    pred_rot_bins = pred_rot_bins.view(bs, num_p, num_rot_bins)
    #rot_bins -> bs * num_rot_bins * num_p
    rot_bins = rot_bins.view(bs, 1, num_rot_bins).repeat(1, num_p, 1)

    #t -> bs * num_p * 3
    t = t.repeat(1, num_p, 1)

    #pred_front loss (L2 norm on front vector)
    pred_front_dis = torch.norm((pred_front - (front_r + t)), dim=2).unsqueeze(-1)

    #pred_rot loss (cross entropy on bins)
    pred_rot_loss = cross_entropy_loss(pred_rot_bins.transpose(2, 1), rot_bins.transpose(2, 1)).unsqueeze(-1)

    #pred_t loss (L2 norm on translation)
    pred_t_loss = torch.norm((pred_t - t), dim=2).unsqueeze(-1)

    #print("shapes before loss calc", pred_front_dis.shape, pred_rot_loss.shape, pred_t_loss.shape, pred_c.shape)

    #pred_front_dis -> bs * num_p * 1
    #pred_rot_loss -> bs * num_p * 1
    #pred_t_loss -> bs * num_p * 1
    #pred_c -> bs * num_p * 1
    
    loss = torch.mean(pred_front_dis * front_loss_coeff + pred_rot_loss * rot_bins_loss_coeff + pred_t_loss * translation_loss_coeff)
    #loss = torch.mean(pred_front_dis * front_loss_coeff + pred_t_loss * translation_loss_coeff)
    
    return loss, torch.mean(pred_front_dis), torch.mean(pred_rot_loss), torch.mean(pred_t_loss)


class Loss(_Loss):

    def __init__(self, num_rot_bins):
        super(Loss, self).__init__(True)
        self.num_rot_bins = num_rot_bins

    def forward(self, pred_front, pred_rot_bins, pred_t, front_r, rot_bins, t):
        return loss_calculation(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, t, self.num_rot_bins)
