import os
import glob
from shutil import copy
import sys
import cv2

input_dir = r'/home/LYX/Datasets/NTIRE2024/Track1/Train/LR_x4'
output_dir = r'/home/LYX/Datasets/NTIRE2024/Track1/Train/lr'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

img_list_l = sorted(glob.glob(os.path.join(input_dir, '*_L.png')))
img_list_r = sorted(glob.glob(os.path.join(input_dir, '*_R.png')))

for idx, file in enumerate(img_list_l):
    save_dir = os.path.join(output_dir, str(idx+1))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    copy(file, os.path.join(save_dir, 'lr0.png'))
    copy(img_list_r[idx], os.path.join(save_dir, 'lr1.png'))