## 为了生成testing raw data的窗宽窗位改变之后的，归一化之后的npy文件，为了可以更好的evaluate还有相对应的mask npy文件可以盖住
## 这里都不用调换第一通道和第三通道的位置

# 造高斯函数
import numpy as np
import torchio as tio
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import numpy as np
import os
import csv
import torch
import glob
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from pathlib import Path
import random
import shutil
#* trans the data to the test
import pandas as pd


def generate_label(data_root_path, part, name, number):

    img_path = data_root_path.joinpath(part).joinpath(name)
    file_name = img_path.iterdir() # 迭代器不能够去进行索引
    file_name = list(file_name)
    if len(file_name) == 0:
        print(f'the part : {part}, the name : {name} , have no data!!!!!!!!')
    else:
        img = tio.ScalarImage(os.path.join(img_path, file_name[0]))
        # * 窗宽窗位设置一下
        clamped = tio.Clamp(out_min=-160., out_max=240.)
        clamped_img = clamped(img)
        # * resample一下
        resample = tio.Resample((0.78125, 0.78125, 1.0))
        clamped_img = resample(clamped_img)
        # * 归一化到 0-1 之间
        data_max = clamped_img.data.max()
        data_min = clamped_img.data.min()
        norm_data = (clamped_img.data - data_min) / (data_max - data_min)
        shape = clamped_img.shape[1:]
        np.save(f'/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/testing_npy/lymphdet_{number}.npy' , np.array(norm_data))
        # norm_data = tio.ScalarImage(tensor=norm_data, affine=clamped_img.affine)
        # norm_data.save(f'/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/testing_npy/lymphdet_{number}.nii.gz')



def generate_mask(data_root_path, part, name, number):
        mask_nii = tio.ScalarImage(f"/public_bme/data/xiongjl/lymph_nodes/{part}_mask/{name}/mediastinum.nii.gz")
        # mask_nii = tio.ScalarImage(f"{self._config['lymph_nodes_data_path']}{part}_mask/{name}/mediastinum.nii.gz")
        resample = tio.Resample((0.78125, 0.78125, 1.0))
        mask_nii = resample(mask_nii)
        mask_data = mask_nii.data.unsqueeze(0)
        dilated_mask = binary_dilation(mask_data, iterations=10)
        np.save(f'/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/mask_npy/lymphdet_mask_{number}.npy', dilated_mask)


if __name__ == '__main__':

    data_root_path = Path('/public_bme/data/xiongjl/lymph_nodes/raw_data/')
    parts = ['testing']

    # with open('/public_bme/data/xiongjl/nnDet/csv_files/name2number.txt', 'a') as f:
    for part in parts:
        names_list = []
        with open(f'/public_bme/data/xiongjl/nnDet/csv_files/{part}_names.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                names_list.append(row[0])

        for i, name in tqdm(enumerate(names_list)):
            if part == 'training':
                number = str(i)
            if part == 'validation':
                number = str(i+870)
            if part == 'testing':
                number = str(i+970)
            if len(number) == 1:
                number = f'00{number}'
            elif len(number) == 2:
                number = f'0{number}'
            elif len(number) == 3:
                pass
            # else:
                # print(f'the number given wrong, now is {number}')
            # f.write(f'{name}:{number}:{part}\n')
            generate_mask(data_root_path, part, name, number)


