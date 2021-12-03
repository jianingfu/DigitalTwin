import os
import random
random.seed(123)

path = "data/1"
files = [f.split('.')[0] for f in os.listdir(path) if f.find("left.png") != -1]
total_data_num = len(files)
print("There are total {} data".format(total_data_num))

ratio = 0.8
random.shuffle(files)
train_data = files[:  int(total_data_num * ratio)]
test_data = files[int(total_data_num * ratio):]
print("Split into {} training data, {} testing data".format(len(train_data), len(test_data)))

train_output_path = "train.txt"
with open(train_output_path, "w") as textfile:
    for data in train_data:
        textfile.write(data + "\n")
print("Write train data into file: {}".format(train_output_path))

test_output_path = "test.txt"
with open(test_output_path, "w") as textfile:
    for data in test_data:
        textfile.write(data + "\n")
print("Write test data into file: {}".format(test_output_path))



