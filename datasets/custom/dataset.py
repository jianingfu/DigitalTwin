import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2
import open3d as o3d

def standardize_image_size(target_image_size, rmin, rmax, cmin, cmax, image_height, image_width):
    height, width = rmax - rmin, cmax - cmin

    if height > target_image_size:
        diff = height - target_image_size
        rmin += int(diff / 2)
        rmax -= int((diff + 1) / 2)
    
    elif height < target_image_size:
        diff = target_image_size - height
        if rmin - int(diff / 2) < 0:
            rmax += diff
        elif rmax + int((diff + 1) / 2) >= image_height:
            rmin -= diff
        else:
            rmin -= int(diff / 2)
            rmax += int((diff + 1) / 2)
    
    if width > target_image_size:
        diff = width - target_image_size
        cmin += int(diff / 2)
        cmax -= int((diff + 1) / 2)
    
    elif width < target_image_size:
        diff = target_image_size - width
        if cmin - int(diff / 2) < 0:
            cmax += diff
        elif cmax + int((diff + 1) / 2) >= image_width:
            cmin -= diff
        else:
            cmin -= int(diff / 2)
            cmax += int((diff + 1) / 2)
    
    return rmin, rmax, cmin, cmax
    

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine, image_size, cropped):
        self.objlist = [1]
        self.mode = mode

        self.cropped = cropped

        self.data_list = []
        self.model_list = []
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.object_number = []

        for item in self.objlist:

            print("Loading Object {0} buffer".format(item))
                           
            if self.mode == 'train':
                input_file = open('{}/split/{}/train.txt'.format(self.root, item))
            else:
                input_file = open('{}/split/{}/test.txt'.format(self.root, item))
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                if cropped:
                    self.data_list.append('{}/data/{}_cropped/{}_cropped/{}'.format(self.root, item, mode, input_line))
                else:
                    self.data_list.append('{}/data/{}/{}'.format(self.root, item, input_line))
                self.object_number.append(item)

            print("Object {0} buffer loaded".format(item))

        self.length = len(self.data_list)
        print("Dataset Length: {}".format(self.length))

        with open("{}/data/{}/_camera_settings.json".format(self.root, self.objlist[0])) as cam_config:
            cam_data = json.load(cam_config)
            self.cam_cx = float(cam_data["camera_settings"][0]["intrinsic_settings"]["cx"])
            self.cam_cy = float(cam_data["camera_settings"][0]["intrinsic_settings"]["cy"])
            self.cam_fx = float(cam_data["camera_settings"][0]["intrinsic_settings"]["fx"])
            self.cam_fy = float(cam_data["camera_settings"][0]["intrinsic_settings"]["fy"])

        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])
        
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = []

        self.image_size = image_size

    def __getitem__(self, index):
        path_prefix = self.data_list[index]
        object_id = self.object_number[index]
        model_path = "{}/models/{}/1_centered.obj".format(self.root, object_id)

        img = Image.open(path_prefix + ".left.png")

        depth = np.array(Image.open(path_prefix + ".left.depth.16.png"))
        label = np.array(Image.open(path_prefix + ".left.cs.png"))
        with open(path_prefix + ".left.json") as cam_config:
            meta = json.load(cam_config)

        with open("{}/data/{}/_object_settings.json".format(self.root, object_id)) as object_config:
            object_data = json.load(object_config)
            object_class_id = object_data["exported_objects"][0]["segmentation_class_id"]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(object_class_id)))
        # FIXME: not sure about the else section
        '''
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        '''
        
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3] # remove alpha channel
        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        if self.cropped:
            top_left = np.load(path_prefix + ".left.cropinfo.npy")
            top_left_row, top_left_col = top_left[0], top_left[1]
            rmin, rmax, cmin, cmax = top_left_row, top_left_row + img.shape[1], top_left_col, top_left_col + img.shape[2]

            choose = mask.flatten().nonzero()[0]
        else:
            bbox = meta["objects"][0]["bounding_box"]
            rmin, rmax, cmin, cmax = int(bbox["top_left"][0]), int(bbox["bottom_right"][0]), int(bbox["top_left"][1]), int(bbox["bottom_right"][1])

            _, h, w = img_masked.shape
            rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)
            rmin, rmax, cmin, cmax = standardize_image_size(self.image_size, rmin, rmax, cmin, cmax, img.shape[1], img.shape[2])
        
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            # the indices that is non-zero
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        
        if self.cropped:
            depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        else:
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        choose = np.array([choose])

        cam_scale = 1.0

        #applying siming's depth fix
        pt2 = depth_masked / cam_scale / 65535.0 * 10
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud

        #model_points = ply_vtx(model_path)
        model_points = o3d.io.read_triangle_mesh(model_path)
        model_points = np.array(model_points.vertices)

        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        target_transform = np.array(meta['objects'][0]['pose_transform'])
        target_rotation = target_transform[:3, :3]
        p = np.array([[ 0, 0, 1],
                      [ 1, 0, 0],
                      [ 0,-1, 0]])
        real_target_rotation = np.matmul(target_rotation.T, p)
        target_translation = target_transform[3,:3]

        fixed_transform = np.array(object_data['exported_objects'][0]['fixed_model_transform'])
        fixed_rotation = fixed_transform[:3, :3].T.astype(np.float64) / 100
        fixed_translation = fixed_transform[3,:3].astype(np.float64) / 100

        target = np.dot(model_points, fixed_rotation.T)#+ fixed_translation
        target = np.dot(target, real_target_rotation.T) + target_translation / 100

        img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               img_masked, \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([int(object_id) - 1])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

