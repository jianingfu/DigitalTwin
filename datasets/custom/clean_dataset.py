import os
import json
import numpy as np
from PIL import Image
import numpy.ma as ma
from dataset import standardize_image_size

root = "custom_preprocessed"

target_image_size = 20

objs = [1]

h, w = 720, 1280

def clean_idxs(idx_file, item):
    clean_idxs = []
    
    counter = 0
    while 1:
        input_line = idx_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        prefix = '{}/data/{}/{}'.format(root, item, input_line)

        with open(prefix + ".left.json") as cam_config:
            meta = json.load(cam_config)

            #remove samples with no object
            if len(meta['objects']) == 0:
                print("no obj", prefix)
                continue

            bbox = meta["objects"][0]["bounding_box"]
            rmin, rmax, cmin, cmax = int(bbox["top_left"][0]), int(bbox["bottom_right"][0]), int(bbox["top_left"][1]), int(bbox["bottom_right"][1])
            rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)

            rmin, rmax, cmin, cmax = standardize_image_size(target_image_size, rmin, rmax, cmin, cmax, h, w)

            #IMAGE PATCH must be at least 4x4
            if (rmin + 3 >= rmax):
                print("r small", prefix)
                continue

            if (cmin + 3 >= cmax):
                print("c small", prefix)
                continue

            depth = np.array(Image.open(prefix + ".left.depth.16.png"))
            label = np.array(Image.open(prefix + ".left.cs.png"))

            with open("{}/data/{}/_object_settings.json".format(root, item)) as object_config:
                object_data = json.load(object_config)
                object_class_id = object_data["exported_objects"][0]["segmentation_class_id"]

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(object_class_id)))
            
            mask = mask_label * mask_depth
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if (len(choose) == 0):
                print("choose 0", prefix)
                continue
            
            clean_idxs.append(input_line)
            counter += 1

            if counter % 1000 == 0:
                print("{0} clean samples".format(counter))
            

    return clean_idxs

for item in objs:
    train_file = open('{}/split/{}/train.txt'.format(root, item))
    test_file = open('{}/split/{}/test.txt'.format(root, item))

    clean_train_idxs = clean_idxs(train_file, item)

    clean_test_idxs = clean_idxs(test_file, item)

    print(len(clean_train_idxs), len(clean_test_idxs))

    train_file.close()
    test_file.close()

    with open("train_clean.txt", "w") as f:
        for clean_train_idx in clean_train_idxs:
            f.write(clean_train_idx + "\n")

    with open("test_clean.txt", "w") as f:
        for clean_test_idx in clean_test_idxs:
            f.write(clean_test_idx + "\n")

    
        