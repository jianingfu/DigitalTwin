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
    def __init__(self, mode, num, add_noise, root, noise_trans, refine, image_size):
        self.objlist = [1]
        self.mode = mode

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

    def crop(self, index, output_dir):
        path_prefix = self.data_list[index]
        object_id = self.object_number[index]
        model_path = "{}/models/{}/1_centered.obj".format(self.root, object_id)

        img = Image.open(path_prefix + ".left.png")

        depth = Image.open(path_prefix + ".left.depth.16.png")
        label = Image.open(path_prefix + ".left.cs.png")
        with open(path_prefix + ".left.json") as cam_config:
            meta = json.load(cam_config)

        with open("{}/data/{}/_object_settings.json".format(self.root, object_id)) as object_config:
            object_data = json.load(object_config)
            object_class_id = object_data["exported_objects"][0]["segmentation_class_id"]

        img_arr = np.array(img)

        bbox = meta["objects"][0]["bounding_box"]
        rmin, rmax, cmin, cmax = int(bbox["top_left"][0]), int(bbox["bottom_right"][0]), int(bbox["top_left"][1]), int(bbox["bottom_right"][1])

        h, w, _ = img_arr.shape
        rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)
        rmin, rmax, cmin, cmax = standardize_image_size(self.image_size, rmin, rmax, cmin, cmax, img_arr.shape[0], img_arr.shape[1])

        sample_num = path_prefix.split("/")[-1]

        output_img = os.path.join(output_dir, sample_num + ".left.png")
        output_depth = os.path.join(output_dir, sample_num + ".left.depth.16.png")
        output_label = os.path.join(output_dir, sample_num + ".left.cs.png")
        output_meta = os.path.join(output_dir, sample_num + ".left.json")
        output_crop_info = os.path.join(output_dir, sample_num + ".left.cropinfo.npy")

        img = np.array(img)[rmin:rmax,cmin:cmax]
        depth = np.array(depth)[rmin:rmax,cmin:cmax]
        label = np.array(label)[rmin:rmax,cmin:cmax]

        print("shapes", img.shape, depth.shape, label.shape)

        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        label = Image.fromarray(label)

        img.save(output_img)
        depth.save(output_depth)
        label.save(output_label)

        crop_info = np.array([rmin, cmin])
        np.save(output_crop_info, crop_info)

        with open(output_meta, "w") as f:
            json.dump(meta, f)


    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

def main():
    root = "custom_preprocessed"
    train_data_output = "train_cropped/"
    test_data_output = "test_cropped/"

    if not os.path.isdir(train_data_output):
        os.mkdir(train_data_output)

    if not os.path.isdir(test_data_output):
        os.mkdir(test_data_output)

    image_size = 100

    train_dataset = PoseDataset("train", 50, True, root, 0.03, False, image_size)

    for train_sample in range(train_dataset.length):
        train_dataset.crop(train_sample, train_data_output)
        if train_sample % 100 == 0:
            print("train samples so far", train_sample)

    test_dataset = PoseDataset("test", 50, False, root, 0.03, False, image_size)

    for test_sample in range(test_dataset.length):
        test_dataset.crop(test_sample, test_data_output)
        if test_sample % 100 == 0:
            print("test samples so far", test_sample)

if __name__ == "__main__":
    main()