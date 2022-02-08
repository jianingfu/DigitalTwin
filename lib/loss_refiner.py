from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn

from lib.transformations import rotation_matrix_from_vectors, rotation_matrix_of_axis_angle

cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

rot_bins_loss_coeff = 1
front_loss_coeff = 1
translation_loss_coeff = 4


def loss_calculation(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, front_orig, t, idx, points, num_rot_bins):
    
    pred_front = pred_front.view((1, 1, 3))
    pred_rot_bins = pred_rot_bins.view((1, 1, num_rot_bins))
    pred_t = pred_t.view((1, 1, 3))

    bs, num_p, _ = pred_front.size()
    num_input_points = len(points[0])
    
    front_r = front_r.view(bs, 1, 3).repeat(1, bs*num_p, 1)

    pred_rot_bins = pred_rot_bins.view(bs*num_p, num_rot_bins)
    rot_bins = rot_bins.repeat(bs*num_p, 1)

    #pred_front loss (L2 norm on front vector)
    pred_front_dis = torch.norm((pred_front - front_r), dim=2)

    #pred_rot loss (cross entropy on bins)
    pred_rot_loss = cross_entropy_loss(pred_rot_bins, rot_bins).unsqueeze(0).unsqueeze(-1)

    t = t.repeat(1, bs*num_p, 1)

    #pred_t loss (L2 norm on translation)
    pred_t_loss = torch.norm((pred_t - t), dim=2).unsqueeze(-1)

    loss = torch.mean(pred_front_dis * front_loss_coeff + pred_rot_loss * rot_bins_loss_coeff + pred_t_loss * translation_loss_coeff)

    #calculating new model_points for refiner
    #requires finding highest confidence front and theta and solving for rotation matrix

    pred_t = pred_t[0]
    points = points.view(1, bs * num_input_points, 3)

    #we need to calculate the actual transformation that our rotation rep. represents

    pred_front = pred_front.view(bs * num_p, 3)

    best_c_pred_front = pred_front[0]
    best_c_rot_bins = pred_rot_bins[0]

    front_orig = front_orig.squeeze()

    #calculate actual rotation
    front_orig = front_orig.cpu().detach().numpy()
    best_c_pred_front = best_c_pred_front.cpu().detach().numpy()
    best_c_rot_bins = best_c_rot_bins.cpu().detach().numpy()

    Rf = rotation_matrix_from_vectors(front_orig, best_c_pred_front)

    #get the angle in radians based on highest histogram bin
    angle = np.argmax(best_c_rot_bins) / best_c_rot_bins.shape[0] * 2 * np.pi

    R_axis = rotation_matrix_of_axis_angle(best_c_pred_front, angle)

    R_tot = (R_axis @ Rf).T

    R_tot = torch.from_numpy(R_tot.astype(np.float32)).cuda().contiguous().view(bs, 3, 3)
    pred_t = pred_t.repeat(bs * num_input_points, 1).contiguous().view(1, bs * num_input_points, 3)

    new_points = torch.bmm((points - pred_t), R_tot).contiguous().detach()

    with torch.no_grad():
        new_rot_bins = rot_bins[0]
        new_rot_bins = torch.roll(new_rot_bins, -np.argmax(best_c_rot_bins)).unsqueeze(0)

    with torch.no_grad():
        new_t = torch.unsqueeze(t[:,0,:] - pred_t[:,0,:], 1)

    # # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    return loss, new_points, new_rot_bins, new_t, pred_front_dis * front_loss_coeff, pred_rot_loss * rot_bins_loss_coeff, pred_t_loss * translation_loss_coeff


class Loss_refine(_Loss):

    def __init__(self, num_rot_bins):
        super(Loss_refine, self).__init__(True)
        self.num_rot_bins = num_rot_bins


    def forward(self, pred_front, pred_rot_bins, pred_t, front_r, rot_bins, front_orig, t, idx, points):
        return loss_calculation(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, front_orig, t, idx, points, self.num_rot_bins)

