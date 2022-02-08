from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn

from lib.transformations import rotation_matrix_from_vectors_procedure_batch, rotation_matrix_of_axis_angle_batch
from lib.loss_helper import roll_by_gather, rot_bins_loss_coeff, front_loss_coeff, translation_loss_coeff

cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

def loss_calculation(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, front_orig, t, idx, points, num_rot_bins):
    
    #print("shapes loss refiner", pred_front.shape, pred_rot_bins.shape, pred_t.shape, front_r.shape, rot_bins.shape, front_orig.shape, t.shape, points.shape)

    #pred_front -> bs * 1 * 3
    #pred_rot_bins -> bs * 1 * num_rot_bins
    #pred_t -> bs * 1 * 3

    #front_r -> bs * 3
    #rot_bins -> bs * num_rot_bins
    #front_orig -> bs * 3
    #t -> bs * 1 * 3
    #points -> bs * num_p * 3

    #orig_rot_bins -> bs * num_rot_bins
    #used for new_rot_bins calculation
    orig_rot_bins = rot_bins

    bs, _, _ = pred_front.size()
    num_input_points = len(points[0])
    
    #front_r -> bs * 1 * 3
    front_r = front_r.view(bs, 1, 3)

    #rot_bins -> bs * 1 * num_rot_bins
    rot_bins = rot_bins.view(bs, 1, num_rot_bins)

    #pred_front loss (L2 norm on front vector)
    pred_front_dis = torch.norm((pred_front - front_r), dim=2).unsqueeze(-1)

    #pred_rot loss (cross entropy on bins)
    pred_rot_loss = cross_entropy_loss(pred_rot_bins.transpose(2, 1), rot_bins.transpose(2, 1)).unsqueeze(-1)

    #pred_t loss (L2 norm on translation)
    pred_t_loss = torch.norm((pred_t - t), dim=2).unsqueeze(-1)

    #print("shapes before loss calc refiner", pred_front_dis.shape, pred_rot_loss.shape, pred_t_loss.shape)

    #pred_front_dis -> bs * 1 * 1
    #pred_rot_loss -> bs * 1 * 1
    #pred_t_loss -> bs * 1 * 1

    loss = torch.mean(pred_front_dis * front_loss_coeff + pred_rot_loss * rot_bins_loss_coeff + pred_t_loss * translation_loss_coeff)

    #calculating new model_points for refiner

    #we need to calculate the actual transformation that our rotation rep. represents

    best_c_pred_t = pred_t
    best_c_pred_front = pred_front.view(bs, 3)
    best_c_rot_bins = pred_rot_bins.view(bs, num_rot_bins)

    #calculate actual rotation
    front_orig = front_orig.cpu().detach().numpy()
    best_c_pred_front = best_c_pred_front.cpu().detach().numpy()
    best_c_rot_bins = best_c_rot_bins.cpu().detach().numpy()

    #Rf -> bs * 3 * 3
    Rf = rotation_matrix_from_vectors_procedure_batch(front_orig, best_c_pred_front)

    #get the angle in radians based on highest histogram bin
    #angle -> bs * 1
    angle = np.expand_dims(np.argmax(best_c_rot_bins, axis=1) / best_c_rot_bins.shape[1] * 2 * np.pi, axis=-1)

    R_axis = rotation_matrix_of_axis_angle_batch(best_c_pred_front, angle)

    #R_tot -> bs * 3 * 3
    #transposed since it will be right multiplied
    R_tot = np.matmul(R_axis, Rf).transpose(0, 2, 1)

    R_tot = torch.from_numpy(R_tot.astype(np.float32)).cuda().contiguous().view(bs, 3, 3)
    best_c_pred_t = best_c_pred_t.view(bs, 1, 3).repeat(1, num_input_points, 1)

    #new_points -> bs * num_p * 3
    new_points = torch.bmm((points - best_c_pred_t), R_tot).contiguous().detach()

    shifts = -np.argmax(best_c_rot_bins, axis=1)
    shifts = torch.from_numpy(shifts).type(torch.LongTensor).view(bs, 1).cuda()

    with torch.no_grad():
        #new_rot_bins -> bs * num_rot_bins
        new_rot_bins = roll_by_gather(orig_rot_bins, 1, shifts)

    with torch.no_grad():
        new_t = torch.unsqueeze(t[:,0,:] - best_c_pred_t[:,0,:], 1)

    # # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    return loss, new_points, new_rot_bins, new_t


class Loss_refine(_Loss):

    def __init__(self, num_rot_bins):
        super(Loss_refine, self).__init__(True)
        self.num_rot_bins = num_rot_bins


    def forward(self, pred_front, pred_rot_bins, pred_t, front_r, rot_bins, front_orig, t, idx, points):
        return loss_calculation(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, front_orig, t, idx, points, self.num_rot_bins)
