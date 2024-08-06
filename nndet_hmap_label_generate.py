
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



def create_gaussian_base(size, threshold):

    if size <= 9:
        _size = 9
        half_dis = (_size + 1) / 2.
    else:
        _size = size
        if _size % 2 != 1:  # 如果size是偶数就变成奇数
            half_dis = _size / 2.
            _size = _size + 1
        else:
            half_dis = (_size + 1) / 2.

    if threshold == 0.5:
        sigma = np.sqrt(half_dis**2 / (2 * np.log(2)))
    elif threshold == 0.8:
        sigma = np.sqrt(half_dis**2 / (2 * (np.log(5) - np.log(4))))
    elif threshold == 0.3:
        sigma = np.sqrt(half_dis**2 / (2 * (np.log(10) - np.log(3))))
    else:
        print(f'when x = distance, the y wrong input, now the threshold is {threshold}')

    kernel = np.zeros((int(_size), int(_size), int(_size)))
    center = tuple(s // 2 for s in (int(_size), int(_size), int(_size)))
    kernel[center] = 1
    gassian_kernel = gaussian_filter(kernel, sigma=sigma)

    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间
    # print(f'in the create_gaussian_base , the max is {normalized_arr.max()}, the min is {normalized_arr.min()}')
    return normalized_arr


def create_gaussian_kernel_v5(whd):
    # 定义新的维度
    new_dims_w = int(whd[0])   # 新的长方体的维度
    new_dims_h = int(whd[1])
    new_dims_d = int(whd[2])
    size_max = int(np.max(whd))


    if new_dims_w % 2 == 0:
        new_dims_w += 1
    if new_dims_h % 2 == 0:
        new_dims_h += 1
    if new_dims_d % 2 == 0:
        new_dims_d += 1
    if size_max % 2 == 0:
        size_max += 1

    new_w = new_dims_w / size_max
    new_h = new_dims_h / size_max
    new_d = new_dims_d / size_max

    gaussian_kernel = create_gaussian_base(size_max, 0.3)
    # 使用scipy.ndimage.zoom函数来伸缩高斯核
    # rescaled_kernel = zoom(gaussian_kernel, (new_dims_w, new_dims_h, new_dims_d))
    rescaled_kernel = zoom(gaussian_kernel, (new_w, new_h, new_d))
    rescaleded_kernel = add_dim_inarray(rescaled_kernel)
    # print(f'rescaled_kernel.shape is {rescaled_kernel.shape}, and the rescaleded_kernel.shape is {rescaleded_kernel.shape}')

    return rescaleded_kernel


def add_dim_inarray(array):
    shape = np.shape(array)
    w, h, d = shape

    if w % 2 == 0:
        w += 1
        new_array = np.ones((w, h, d))
        # print((w-1)/2+ 1)
        new_array[0:int((w-1)/2 + 1), :, :] = array[0:int((w-1)/2 + 1), :, :]
        new_array[int((w-1)/2 + 1), :, :] = array[int((w-1)/2 ), :, :]
        new_array[int((w-1)/2 + 2) : w+1 , :, :] = array[int((w-1)/2 + 1) : w, :, :]
        array = new_array
    if h % 2 == 0:
        h += 1
        new_array = np.ones((w, h, d))
        new_array[:, 0:int((h-1)/2 + 1), :] = array[:, 0:int((h-1)/2 + 1), :]
        new_array[:, int((h-1)/2 + 1), :] = array[:, int((h-1)/2 ), :]
        new_array[:, int((h-1)/2 + 2) : w+1 , :] = array[:, int((h-1)/2 + 1) : h, :]
        array = new_array
    if d % 2 == 0:
        d += 1
        new_array = np.ones((w, h, d))
        new_array[:, :, 0:int((d-1)/2 + 1)] = array[:, :, 0:int((d-1)/2 + 1)]
        new_array[:, :, int((d-1)/2 + 1)] = array[:, :, int((d-1)/2 )]
        new_array[:, :, int((d-1)/2 + 2) : d+1 ] = array[:, :, int((d-1)/2 + 1) : d]
        array = new_array

    return array


def place_gaussian(arr, kernel, pos):
    x, y, z = pos
    kx, ky, kz = kernel.shape
    # 计算高斯核在数组中的位置
    x1, x2 = max(0, x-kx//2), min(arr.shape[0], x+kx//2+1)
    y1, y2 = max(0, y-ky//2), min(arr.shape[1], y+ky//2+1)
    z1, z2 = max(0, z-kz//2), min(arr.shape[2], z+kz//2+1)
    # 计算高斯核在自身中的位置
    kx1, kx2 = max(0, kx//2-x), min(kx, kx//2-x+arr.shape[0])
    ky1, ky2 = max(0, ky//2-y), min(ky, ky//2-y+arr.shape[1])
    kz1, kz2 = max(0, kz//2-z), min(kz, kz//2-z+arr.shape[2])
    # 将高斯核放置在指定位置
    arr[x1:x2,y1:y2,z1:z2] = np.maximum(arr[x1:x2,y1:y2,z1:z2], kernel[kx1:kx2,ky1:ky2,kz1:kz2])

    return arr


def create_hmap_v5(coordinates, shape):
    arr = np.zeros(shape)
    for coords in coordinates:
        coord = [int(x) for x in coords[0:3]]
        whd = [int(x) for x in coords[3:6]]
        kernel = create_gaussian_kernel_v5(whd)
        arr = place_gaussian(arr, kernel, coord)

    return arr


def generate_label(data_root_path, part, name, number):

    img_path = data_root_path.joinpath(part).joinpath(name)
    file_name = img_path.iterdir() # 迭代器不能够去进行索引
    file_name = list(file_name)
    if len(file_name) == 0:
        print(f'the part : {part}, the name : {name} , have no data!!!!!!!!')
    else:
        img = tio.ScalarImage(os.path.join(img_path, file_name[0]))
        source = os.path.join(img_path, file_name[0])
        # destination = f'D:\\Work_file\\uii_lymph_nodes_data\\DATASET\\testingTr\\lymph_{number}_0000.nii.gz'
        # shutil.copy(source, destination)
        # img.save(f'D:\\Work_file\\uii_lymph_nodes_data\\DATASET\\imagesTr\\lymph_{number}_0000.nii.gz')
        # * 读取csv文件中的世界坐标/public_bme/data/xiongjl/nnDet/csv_files/CTA_thin_std_testing_lymph_refine.csv
        worldcoord = pd.read_csv(f'/public_bme/data/xiongjl/nnDet/csv_files/CTA_thin_std_{part}_lymph_refine.csv')
        # csv_filename = f'{data_root_path}/lymph_csv_refine/{part}_npyrefine.csv'
        csv_filename = f'/public_bme/data/xiongjl/lymph_det/csv_files/{part}_npyrefine.csv'
        raw = worldcoord[worldcoord['image_path'].str.contains(name)]
        coords = []
        for i in range(len(raw)):
            x = raw.iloc[i, 2]
            y = raw.iloc[i, 3]
            z = raw.iloc[i, 4]
            width = raw.iloc[i, 5]
            height = raw.iloc[i, 6]
            depth = raw.iloc[i, 7]
            coords.append([x, y, z, width, height, depth]) # 这个是世界坐标系
        # print(f'the world coords is {coords}')

        # * 把世界坐标系转化为图像坐标系
        origin = img.origin
        spacing = img.spacing
        shape = img.shape[1:]
        # print(f'the origin is {origin}')
        img_coords = []
        for coord in coords:
            img_coord = (np.array(coord[0:3]) - np.array(origin) * np.array([-1., -1., 1.]) ) / np.array(spacing) # img.spacing
            coord[3: 6] = coord[3: 6] / np.array(spacing)
            img_coords.append([img_coord[0], img_coord[1], img_coord[2], coord[3], coord[4], coord[5]])   #! xyzwhd

        # * 开始生成并且保存这个hmap
        hmap = create_hmap_v5(img_coords, shape)
        # hmap = np.where(hmap >= 0.5, 1, 0)
        hmap_nii = tio.ScalarImage(tensor=torch.tensor(hmap).unsqueeze(0), affine=img.affine)
        hmap_nii.save(f'/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/labelsTr/lymphdet_{number}.nii.gz')

    return hmap



if __name__ == '__main__':

    data_root_path = Path('/public_bme/data/xiongjl/lymph_nodes/raw_data/')
    parts = ['training', 'validation']

    with open('/public_bme/data/xiongjl/nnDet/csv_files/name2number.txt', 'a') as f:
        for part in parts:
            names_list = []
            with open(f'/public_bme/data/xiongjl/nnDet/csv_files/{part}_names.csv') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    names_list.append(row[0])

            if part == 'training':
                for i, name in tqdm(enumerate(names_list)):
                    
                    number = str(i)
                    if len(number) == 1:
                        number = f'00{number}'
                    elif len(number) == 2:
                        number = f'0{number}'
                    elif len(number) == 3:
                        pass
                    else:
                        print(f'the number given wrong, now is {number}')
                    f.write(f'{name}:{number}:{part}\n')
                    hmap = generate_label(data_root_path, part, name, number)

            if part == 'validation':
                for i, name in tqdm(enumerate(names_list)):
                    
                    number = str(i+870)
                    if len(number) == 1:
                        number = f'00{number}'
                    elif len(number) == 2:
                        number = f'0{number}'
                    elif len(number) == 3:
                        pass
                    else:
                        print(f'the number given wrong, now is {number}')
                    f.write(f'{name}:{number}:{part}\n')
                    hmap = generate_label(data_root_path, part, name, number)




