import csv
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torch
from torch import nn
import csv
import torch.nn.functional as F
import os
from tqdm import tqdm
from time import time
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import torchio as tio
from scipy.ndimage import binary_dilation
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#* the test is for window crop for a image
from pathlib import Path
from time import time

class DetectionEvaluator:
    def __init__(self, config) -> None:
        self._config = config
        self._patch_size = config["patch_size"]
        self._overlap = config["overlap"]
        self._confidence = config["confidence"]

    def __call__(self, model, number_epoch, timesteamp, fold):
        print(f'>start to Evaluation...{number_epoch}')
        
        path = [
            '/public_bme/data/xiongjl/nnDet/csv_files/part_testing_trainingnames.csv',
            '/public_bme/data/xiongjl/nnDet/csv_files/part_testing_names.csv',
        ]
        # to get the names of the training or the testing
        txt_paths = []
        data_root_path = Path('/public_bme/data/xiongjl/lymph_nodes/raw_data/')
        for file_path in path:
            test_names = read_names_from_csv(file_path)
            print(f'the file_path is {file_path}')
            if 'training' in file_path:
                part = 'training'
                det_file = 'train'
            else:
                part = 'testing'
                det_file = 'test'
            txt_path_ = f"{self._config['boxes_save_path']}/{timesteamp}_epoch_{number_epoch}_{det_file}"
            txt_paths.append(txt_path_)
            model.eval()
            scale = [1., 1., 1.]
            step = [pahs - ovlap for pahs, ovlap in zip(self._patch_size, self._overlap)]
            pbar = tqdm(test_names)
            # print(f'part is {part}')
            for name in pbar:
                time_s = time()
                pbar.set_description('Evaluation')
                csv_path = '/public_bme/data/xiongjl/nnDet/csv_files/nnunet_data_output.csv'
                # name = '02200422216091' #* 4. xjl add for the overfit
                label_xyzwhd, other_data = extract_data_from_csv(name, csv_path)
                # print(f'the labelxyzwhd is {label_xyzwhd}, other_data is {other_data}')
                number = other_data['number']
                # print(f'number is {number}, name is {name}')
                # new_shape = other_data['new_shape']
                path = f"/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/testing_npy/lymphdet_{number}.npy"
                if not os.path.exists(path):
                    print(f'testing npy is not exist')
                    img_path = data_root_path.joinpath(part).joinpath(name)
                    file_name = img_path.iterdir() # 迭代器不能够去进行索引
                    file_name = list(file_name)
                    img = tio.ScalarImage(os.path.join(img_path, file_name[0]))
                    # * 窗宽窗位设置一下
                    clamped = tio.Clamp(out_min=-160., out_max=240.)
                    clamped_img = clamped(img)
                    # * resmaple一下
                    resample = tio.Resample((0.78125, 0.78125, 1.0))
                    clamped_img = resample(clamped_img)
                    # * 归一化到 0-1 之间
                    data_max = clamped_img.data.max()
                    data_min = clamped_img.data.min()
                    norm_data = (clamped_img.data - data_min) / (data_max - data_min)
                    image_data = np.array(norm_data)
                    np.save(path, image_data)
                    # print(f'image data shape is {image_data.shape}')
                else:
                    image_data = np.load(path)
                '''
                img = tio.ScalarImage("/public_bme/data/xiongjl/temp/testingtr_resample07/lymph_099_0000.nii.gz") # xjl add for testing
                # * 窗宽窗位设置一下
                clamped = tio.Clamp(out_min=-160., out_max=240.)
                clamped_img = clamped(img)
                # * 归一化到 0-1 之间
                data_max = clamped_img.data.max()
                data_min = clamped_img.data.min()
                norm_data = (clamped_img.data - data_min) / (data_max - data_min)
                image_data = np.array(norm_data)
                '''
                    # print(f'image data shape is {image_data.shape}')
                # image_data = np.load(f"/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_preprocessed/Dataset502_lymphdet/nnUNetPlans_3d_fullres/lymphdet_{number}.npy")
                # image_data = np.transpose(image_data, (0, 3, 2, 1))
                image_data = image_data[0, :, :, :] # xjl add 这个时候的image的xyz已经normal

                #* the mask 
                mask_path = f"/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/mask_npy/lymphdet_mask_{number}.npy"
                if not os.path.exists(mask_path):
                    print(f'mask path is not exist')
                    mask_nii = tio.ScalarImage(f"/public_bme/data/xiongjl/lymph_nodes/{part}_mask/{name}/mediastinum.nii.gz")
                    resample = tio.Resample((0.78125, 0.78125, 1.0))
                    mask_nii = resample(mask_nii)
                    mask_data = mask_nii.data.unsqueeze(0)
                    mask_data = binary_dilation(mask_data, iterations=10)
                    np.save(mask_path, mask_data)
                else:
                    mask_data = np.load(mask_path) # 5维的并且resample过后的
                mask_data = mask_data[0, 0, :, :, :]
                # mask_data = np.ones(image_data.shape) #* xjl add for testing
                save_shape = mask_data.shape
                # print(f'mask data shape is {mask_data.shape}')
                # mask_data = torch.tensor(mask_data)
                # mask_data = np.array(resample_nii.data.squeeze(0))
                # dilated_mask = binary_dilation(mask_data, iterations=10)
                # print(f'dilated_mask shape is {dilated_mask.shape}')
                image_data, coords_crop = use_mask_and_crop(image_data, mask_data)
                # print(f'coords_crop is {coords_crop}')

                image_patches, arr_pad_shape = sliding_window_3d_volume_padded(image_data, patch_size=self._patch_size, stride=step)
                # print(f'sliding window time is {time() - time_sta}')
                #* the image_patch is a list consist of the all patch of a whole image
                #* each element in the list is a dict consist of start point and tensor(input)
                whole_hmap = np.zeros(arr_pad_shape)
                # whole_hmap_two = np.zeros(arr_pad_shape)
                whole_whd = np.zeros(np.hstack(((3), arr_pad_shape)))
                # time_model = time()
                # print(f'the number of patch is {len(image_patches)}')
                pred_bboxes = []
                for image_patch in image_patches:
                    with torch.no_grad():
                        image_input = image_patch['image'].unsqueeze(0)
                        point = image_patch['point'][1:]
                        order = image_patch['point'][0]
                        image_input = image_input.cuda()
                        image_input = image_input.float()
                        output = model(image_input)
                        output = output[0]
                        pred_hmap = output[:, :1, :, :, :]
                        pred_hmap[pred_hmap > 1] = 1
                        pred_hmap[pred_hmap < 0] = 0
                        pred_whd = output[:, 1:, :, :, :]

                        whole_hmap = place_small_image_in_large_image(whole_hmap, pred_hmap.squeeze(0).squeeze(0).cpu(), point)
                        whole_whd = place_small_image_in_large_image(whole_whd, pred_whd.squeeze(0).cpu(), point)

                (x_min, x_max, y_min, y_max, z_min, z_max) = coords_crop[:]
                dilated_mask = mask_data[x_min:x_max, y_min:y_max, z_min:z_max]
                new_shape = dilated_mask.shape # image_data.shape xjl add 这两者只差一个像素这种
                dilated_mask = torch.tensor(dilated_mask).unsqueeze(0).unsqueeze(0)
                # print(f'coords_crop is {coords_crop}')
                # print(f'after crop, the image data shape is {image_data.shape}, the mask shape is {dilated_mask.shape}')

                # print(f'new_shape is {new_shape}')
                whole_hmap = torch.from_numpy(whole_hmap[0: new_shape[0], 0: new_shape[1], 0: new_shape[2]]).unsqueeze(0).unsqueeze(0)
                # whole_hmap_two = torch.from_numpy(whole_hmap_two[0: new_shape[0], 0: new_shape[1], 0: new_shape[2]]).unsqueeze(0).unsqueeze(0)
                whole_whd = torch.from_numpy(whole_whd[:, 0: new_shape[0], 0: new_shape[1], 0: new_shape[2]]).unsqueeze(0)  
                # print(f'the number is {number}')

                if whole_hmap.shape == dilated_mask.shape:
                    whole_hmap = whole_hmap * dilated_mask
                else:
                    print(f'name:{name}--number:{number}--whole hmap shape : ({whole_hmap.shape}) != mask_data shape : ({dilated_mask.shape})')
                    
                # * 反转第二维度和第四维度的位置,不需要了，因为image进来的时候已经转过了
                pred_bboxes = decode_bbox(self._config, whole_hmap, whole_whd, scale, self._confidence, reduce=1., cuda=True, point=(0,0,0))
                if len(pred_bboxes) >= 150:
                    pred_bboxes = pred_bboxes[0: 150]
                # print(f'the time of bbox is {time() - time_bbox}')

                ground_truth_boxes = centerwhd_2nodes(label_xyzwhd, point=(0, 0, 0))

                # print(f'start nms............ the pred_bboxes is {len(pred_bboxes)}')
                nms_time = time()
                pred_bboxes = nms_(pred_bboxes, thres=self._config['nms_threshold'])
                # print(f'nms time is {time() - time_s}')
                # pred_bboxes = merge_boxes(pred_bboxes, threshold=0.25)

                for bbox in pred_bboxes:
                    hmap_score, x1, y1, z1, x2, y2, z2 = bbox
                    # print(f'hmap_score, x1, y1, z1, x2, y2, z2 is {hmap_score, x1, y1, z1, x2, y2, z2}')
                    txt_path = f"{self._config['boxes_save_path']}3d_fullres_fold_{fold}/{timesteamp}_epoch_{number_epoch}_{det_file}"
                    # txt_path = f"{self._config['boxes_save_path']}/{timesteamp}_epoch_{number_epoch}_{det_file}"
                    if not os.path.exists(txt_path):
                        os.makedirs(txt_path)
                    with open(f"{txt_path}/{name}.txt", 'a') as f:
                        x1 += x_min
                        x2 += x_min
                        y1 += y_min
                        y2 += y_min
                        z1 += z_min
                        z2 += z_min
                        f.write(f'nodule {hmap_score} {x1} {y1} {z1} {x2} {y2} {z2} {str(save_shape).replace(" ", "")}\n')
                # print(f'time of a data is {time() - time_s}')
            np.save('/public_bme/data/xiongjl/nnDet/temp/whole_hmap_infer.npy', torch.tensor(whole_hmap).cpu().detach().numpy())
            np.save('/public_bme/data/xiongjl/nnDet/temp/image_data_infer.npy', torch.tensor(image_data).cpu().detach().numpy())
            print('save the infer thing down!!!!!!!!!!!!!!!!!!')
        return txt_paths
    

def use_mask_and_crop(image, mask):
    coords = np.argwhere(mask)
    # print(coords.min(axis=0))
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1
    cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped_image, (x_min, x_max, y_min, y_max, z_min, z_max)


def get_ap(recall, precision):
    recall_arr = np.array(recall)
    precision_arr = np.array(precision)
    ap_corrected = np.trapz(recall_arr, precision_arr)
    return ap_corrected


def delete_csv_with_best_in_name(det_file, path):
    for filename in os.listdir(path):
        if "best" in filename and filename.endswith(".csv") and det_file in filename:
            os.remove(os.path.join(path, filename))
            print(f"已删除文件: {filename}")



def get_predboxes(pred, confidence):
    result = []
    for pre in pred:
        confi = pre[0]
        if confi >= confidence:
            result.append(pre)
    return result



def filter_boxes(gt, pred, iou_confi):
    result = []
    for box in gt:
        overlap = False
        for p_box in pred:
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                break
        if not overlap:
            result.append(box)

    return result


def get_fp(gt, pred, iou_confi):
    result = []
    for box in pred:
        overlap = False
        for p_box in gt:
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                break
        if not overlap:
            result.append(box)

    return result

def iou(boxA, boxB):
    # if boxes dont intersect
    if _boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = _getIntersectionArea(boxA, boxB)
    union = _getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    if iou < 0:
        iou = - iou
        print('the iou < 0, and i do the iou = - iou')
    # print(f'the iou is {iou}, the interArea is {interArea}, the union is {union}')
    assert iou >= 0
    return iou
    
# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[3]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[3]:
        return False  # boxA is left of boxB
    if boxA[2] > boxB[5]:
        return False  # boxA is left of boxB
    if boxB[2] > boxA[5]:
        return False  # boxA is left of boxB
    if boxA[4] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[4]:
        return False  # boxA is below boxB
    return True
def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1) * (zB - zA + 1)

def _getUnionAreas(boxA, boxB, interArea=None):
    # print(f'the boxa is {boxA}, the boxb is {boxB}')
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    # print(f'the areaa is {area_A}, the areab is {area_B}')
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
        # print(f'the interarea is None, the interarea is {interArea}')
    # print(f'the interarea is None, the interarea is {interArea}')
    # print(f'the iou is {area_A + area_B - interArea}')
    return float(area_A + area_B - interArea)

def _getArea(box):
    return (box[3] - box[0] + 1) * (box[4] - box[1] + 1) * (box[5] - box[2] + 1)


import csv
def parse_list(string):
    # 将形如 "[517 517 295]" 的字符串转换为列表
    return list(map(int, string.strip("[]").split()))

def extract_data_from_csv(name, csv_path):
    bbox_list = []
    other_data = None

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['name'] == name:
                bbox_list.append(eval(row['bbox']))
                if other_data is None:
                    # 只需要从第一行中获取其他数据
                    other_data = {
                        'number': row['number'],
                        'origin': eval(row['origin']),
                        'old_shape': eval(row['old_shape']),
                        'new_shape': parse_list(row['new_shape']),
                        'old_spacing': eval(row['old_spacing'])
                    }

    return bbox_list, other_data


def merge_boxes(boxes, threshold=0.4):
    # 将框按照分数从高到低排序
    boxes = sorted(boxes, key=lambda box: box[0], reverse=True)

    merged_boxes = []
    while boxes:
        # 取出分数最高的框
        max_score_box = boxes.pop(0)

        overlaps = []
        for i, box in enumerate(boxes):
            # 计算重叠部分的体积
            overlap = max(0, min(max_score_box[4], box[4]) - max(max_score_box[1], box[1])) * \
                      max(0, min(max_score_box[5], box[5]) - max(max_score_box[2], box[2])) * \
                      max(0, min(max_score_box[6], box[6]) - max(max_score_box[3], box[3]))
            
            # 计算小框的体积
            volume = min((box[4] - box[1]) * (box[5] - box[2]) * (box[6] - box[3]), \
                        (max_score_box[4] - max_score_box[1]) * (max_score_box[5] - max_score_box[2]) * (max_score_box[6] - max_score_box[3]))

            # 如果重叠部分占小框体积的比例大于阈值，则记录下来
            if overlap / volume > threshold:
                overlaps.append(i)

        # 如果有重叠的框，将它们与最大分数框合并
        if overlaps:
            for i in sorted(overlaps, reverse=True):
                overlap_box = boxes.pop(i)
                max_score_box = [max_score_box[0]] + \
                                [min(max_score_box[j+1], overlap_box[j+1]) for j in range(3)] + \
                                [max(max_score_box[j+4], overlap_box[j+4]) for j in range(3)]

        merged_boxes.append(max_score_box)

    return merged_boxes


def pool_nms(heat, kernel):
    pad = (kernel - 1) // 2
    if isinstance(heat, np.ndarray):
        heat = torch.from_numpy(heat)
    # print('///////////////////////////////////////////')
    time_nn_func = time()
    if heat.device == 'cuda:0':
        pass
    else:
        heat = heat.cuda()
    # print('---------------------------------------------------')
    hmax = nn.functional.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=pad)
    # print('0000000000000000000000000000000000000')
    heat = heat.cpu()
    hmax = hmax.cpu()
    # print(f'the time of nn_function pool3d is {time() - time_nn_func}')
    # print(f'the device of heat is {heat.device}')
    keep = (hmax == heat).float()
    # print(f'heat shaps is {heat.shape}')
    # print(f'keep shaps is {keep.shape}')
    # print(f'heat  is {heat[:,:,0:4,0:4,0:4]}')
    # print(f'keep  is {keep[:,:,0:4,0:4,0:4]}')
    # print('2222222222222222222222222222222222222222')
    try:
        result = heat * keep
    except Exception as e:
        print(f"An error occurred: {e}")
    result = heat * keep
    return result

from scipy.ndimage import label, find_objects
def find_top_points_in_regions(mask, pred_hms, top_n=3):
    # 确保pred_hms没有多余的批处理维度
    if pred_hms.shape[0] == 1:
        pred_hms = pred_hms.squeeze(0)

    labeled_array, num_features = label(mask)
    slices = find_objects(labeled_array)
    top_points = []

    for slice_ in slices:
        region = pred_hms[slice_]
        # print(region.size)
        k = min(top_n, region.size) - 1
        # 使用np.argpartition找到最大的k个值的索引
        flat_indices = np.argpartition(-region.ravel(), k)[:k+1]
        # 转换回多维索引
        multi_dim_indices = np.unravel_index(flat_indices, region.shape)
        # 获取这些点的实际值
        values = region[multi_dim_indices]
        # 将这些点的索引和值存储起来
        for i, value in enumerate(values):
            # 调整索引以匹配原始pred_hms的维度，这里需要加上slice_的起始位置
            adjusted_indices = tuple(multi_dim_indices[dim][i] + slice_[dim].start for dim in range(len(slice_)))
            top_points.append((adjusted_indices, value))

    return top_points

def decode_bbox(config, pred_hms, pred_whds, scale, confidence, reduce, point, cuda):
    # print(f'the shape of pred_offset is {pred_offsets.shape}')

    time_pool = time()
    # print('====================================')
    pred_hms    = pool_nms(pred_hms, kernel = config['decode_box_kernel_size'])
    
    heat_map    = np.array(pred_hms[0, :, :, :, :])
    # np.save('/public_bme/data/xiongjl/nnDet/temp/whole_heat_map_infer.npy', heat_map)
    pred_whd    = pred_whds[0, :, :, :, :]
    mask = torch.from_numpy(np.where(heat_map > confidence, 1, 0)).squeeze(0) # .bool() # .squeeze(0).bool()
    # np.save('/public_bme/data/xiongjl/nnDet/temp/whole_mask_infer.npy', mask)
    # indices = np.argwhere(mask == 1)
    mask_tensor = mask.bool()  # 转换为Tensor，如果已经是Tensor则不需要
    top_points = find_top_points_in_regions(mask_tensor.numpy(), heat_map)
    # top_points = 
    xyzwhds = []
    hmap_scores = []
    # for i in range(indices.shape[1]):
    #     coord = indices[:, i]
    for point_info in top_points:
        coord, value = point_info
        x = coord[0]
        y = coord[1]
        z = coord[2]
        # hmap_score = heat_map[0, x, y, z]
        # print(f'--x y z -- : {x}, {y}, {z}')
        w = pred_whd[0, x, y, z] / scale[0]
        h = pred_whd[1, x, y, z] / scale[1]
        d = pred_whd[2, x, y, z] / scale[2]

        center = ((x) * reduce, (y) * reduce, (z) * reduce)
        center = [a / b for a, b in zip(center, scale)]

        xyzwhds.append([center[0], center[1], center[2], w, h, d])
        hmap_scores.append(value)
        # print(f'xyzwhds is {xyzwhds}')
    predicted_boxes = centerwhd_2nodes(xyzwhds, point=point, hmap_scores=hmap_scores)


    return predicted_boxes


def pad_image(image, target_size):
    # 计算每个维度需要填充的数量
    padding = [(0, max(0, target_size - size)) for size in image.shape]
    # 使用pad函数进行填充
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    # 返回填充后的图像
    return padded_image


def sliding_window_3d_volume_padded(arr, patch_size, stride, padding_value=0):
    """
    This function takes a 3D numpy array representing a 3D volume and returns a 4D array of patches extracted using a sliding window approach.
    The input array is padded to ensure that its dimensions are divisible by the patch size.
    :param arr: 3D numpy array representing a 3D volume
    :param patch_size: size of the cubic patches to be extracted
    :param stride: stride of the sliding window
    :param padding_value: value to use for padding
    :return: 4D numpy array of shape (num_patches, patch_size, patch_size, patch_size)
    """
    # regular the shape
    if len(arr.shape) != 3:
        arr = arr.squeeze(0)

    patch_size_x = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_z = patch_size[2]

    stride_x = stride[0]
    stride_y = stride[1]
    stride_z = stride[2]

    # Compute the padding size for each dimension
    pad_size_x = (patch_size_x - (arr.shape[0] % patch_size_x)) % patch_size_x
    pad_size_y = (patch_size_y - (arr.shape[1] % patch_size_y)) % patch_size_y
    pad_size_z = (patch_size_z - (arr.shape[2] % patch_size_z)) % patch_size_z

    # Pad the array
    arr_padded = np.pad(arr, ((0, pad_size_x), (0, pad_size_y), (0, pad_size_z)), mode='constant', constant_values=padding_value)

    # Extract patches using a sliding window approach
    patches = []
    order = 0
    for i in range(0, arr_padded.shape[0] - patch_size_x + 1, stride_x):
        for j in range(0, arr_padded.shape[1] - patch_size_y + 1, stride_y):
            for k in range(0, arr_padded.shape[2] - patch_size_z + 1, stride_z):
                patch = arr_padded[i:i + patch_size_x, j:j + patch_size_y, k:k + patch_size_z]
                if isinstance(patch, np.ndarray):
                    patch = torch.from_numpy(patch).unsqueeze(0)
                else:
                    patch = patch.unsqueeze(0)
                start_point = torch.tensor([order, i, j, k])
                add = {'image': patch, 'point': start_point}
                patches.append(add)
                order += 1
    # return np.array(patches)
    return patches, arr_padded.shape


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names


def select_box(predbox, p):
    selected_box = []
    for box in predbox:
        i = box[0]
        if i >= p:
            selected_box.append(box)
    return selected_box


def nms_(dets, thres):
    '''
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
    :param thres: for example 0.5
    :return: the rest ids of dets
    '''
    # print(f'dets is {dets}')
    x1 = [det[1] for det in dets]
    y1 = [det[2] for det in dets]
    z1 = [det[3] for det in dets]
    x2 = [det[4] for det in dets]
    y2 = [det[5] for det in dets]
    z2 = [det[6] for det in dets]
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) * (z2[i] - z1[i]) for i in range(len(x1))]
    scores = [det[0] for det in dets]
    order = order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # print(f'in the nms, the len of dets is {len(dets)}')
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = [max(x1[i], x1[j]) for j in order[1:]]
        xx2 = [min(x2[i], x2[j]) for j in order[1:]]
        yy1 = [max(y1[i], y1[j]) for j in order[1:]]
        yy2 = [min(y2[i], y2[j]) for j in order[1:]]
        zz1 = [max(z1[i], z1[j]) for j in order[1:]]
        zz2 = [min(z2[i], z2[j]) for j in order[1:]]

        w = [max(xx2[i] - xx1[i], 0.0) for i in range(len(xx1))]
        h = [max(yy2[i] - yy1[i], 0.0) for i in range(len(yy1))]
        d = [max(zz2[i] - zz1[i], 0.0) for i in range(len(zz1))]

        inters = [w[i] * h[i] * d[i] for i in range(len(w))]
        unis = [areas[i] + areas[j] - inters[k] for k, j in enumerate(order[1:])]
        ious = [inters[i] / unis[i] for i in range(len(inters))]

        inds = [i for i, val in enumerate(ious) if val <= thres]
         # return the rest boxxes whose iou<=thres

        order = [order[i + 1] for i in inds]

            # inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3
    result = [dets[i] for i in keep]
    # print(f'after the nms, the len of result is {len(result)}')
    return result


def non_overlapping_boxes(boxes):
    non_overlapping = []
    for i, box1 in enumerate(boxes):
        overlapping = False
        for j, box2 in enumerate(non_overlapping):
            if boxes_overlap(box1, box2):
                overlapping = True
                if box_area(box1) > box_area(box2):
                    non_overlapping.remove(box2)
                    non_overlapping.append(box1)
                break
        if not overlapping:
            non_overlapping.append(box1)
    return non_overlapping


def boxes_overlap(box1, box2):
    x1, y1, z1, x2, y2, z2 = [np.float16(x) for x in box1[1:]]
    a1, b1, c1, a2, b2, c2 = [np.float16(x) for x in box2[1:]]
    return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1 or z2 < c1 or c2 < z1)


def box_area(box):
    _, x1, y1, z1, x2, y2, z2 = box
    return (x2 - x1) * (y2 - y1) * (z2 - z1)


def name2coord(config, part, mhd_name):
    # * 输入name，输出这个name所对应着的gt坐标信息
    xyzwhd = []
    if config['data_type'] == 'masked':
        csv_file_dir = f"/public_bme/data/xiongjl/lymph_det/csv_files/{part}_refine_crop.csv"
    else:
        csv_file_dir = f"/public_bme/data/xiongjl/lymph_nodes/lymph_csv_refine/{part}_refine.csv"
    with open(csv_file_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            
            if row[0] == "'" + mhd_name:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                w = float(row[4]) 
                h = float(row[5]) 
                d = float(row[6]) 
                xyzwhd.append((x, y, z, w, h, d))
    return xyzwhd


def place_small_image_in_large_image(large_image, small_image, start_coords):

    if (start_coords[0] < 0 or start_coords[1] < 0 or start_coords[2] < 0 or
            start_coords[0] + small_image.shape[-3] > large_image.shape[-3] or
            start_coords[1] + small_image.shape[-2] > large_image.shape[-2] or
            start_coords[2] + small_image.shape[-1] > large_image.shape[-1]):
        raise ValueError("小图像的起始坐标超出大图像范围")
    
    # 获取小图像的坐标范围
    x_start, y_start, z_start = start_coords
    x_end = x_start + small_image.shape[-3]
    y_end = y_start + small_image.shape[-2]
    z_end = z_start + small_image.shape[-1]
    
    # 将小图像放入大图像中，选择最大值
    if len(large_image.shape) == 3:
        large_image[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            large_image[x_start:x_end, y_start:y_end, z_start:z_end],
            small_image
        )

    elif len(large_image.shape) == 4:
        large_image[:, x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            large_image[:, x_start:x_end, y_start:y_end, z_start:z_end],
            small_image
        )
    else:
        print(f'large image shape should be 3 or 4, but now is {len(large_image.shape)}')
    return large_image


def centerwhd_2nodes(xyzwhds, point, hmap_scores=None):
    if hmap_scores != None:
        result = []
        x_sta, y_sta, z_sta = point
        for xyzwhd, hmap_score in zip(xyzwhds, hmap_scores):

            x, y, z, length, width, height = xyzwhd
            x1 = max(0, x - length/2.0)
            y1 = max(0, y - width/2.0)
            z1 = max(0, z - height/2.0)
            x2 = x + length/2.0
            y2 = y + width/2.0
            z2 = z + height/2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([hmap_score, x1, y1, z1, x2, y2, z2])
        return result
    
    else:
        result = []
        x_sta, y_sta, z_sta = point
        for xyzwhd in xyzwhds:

            x, y, z, length, width, height = xyzwhd
            x1 = max(0, x - length / 2.0)
            y1 = max(0, y - width / 2.0)
            z1 = max(0, z - height / 2.0)
            x2 = x + length / 2.0
            y2 = y + width / 2.0
            z2 = z + height / 2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([x1, y1, z1, x2, y2, z2])

        return result
  

def normal_list(list):
    new_list = []
    for lit in list:
        if lit == []:
            continue
        else:
            for l in lit:
                new_list.append(l)
    return new_list


def cxcyczwhd2x1y1z1x2y2z2(coords): 
    x1y1z1 = []
    for coord in coords:
        # print(coord)
        x, y, z, w, h, d = coord
        x1 = x - w / 2. 
        y1 = y - h / 2. 
        z1 = z - d / 2. 
        x2 = x + w / 2. 
        y2 = y + h / 2. 
        z2 = z + d / 2. 
        x1y1z1.append([x1, y1, z1, x2, y2, z2])

    return x1y1z1



def process_nii_image(data, bboxes, output_path):
    data = data[0, :, :, :]
    # print(bboxes)
    # 遍历每个框
    bboxes = cxcyczwhd2x1y1z1x2y2z2(bboxes)
    for bbox in bboxes:
        
        x1, y1, z1, x2, y2, z2 = map(int, bbox)
        # print(x1, y1, z1, x2, y2, z2)
        # 将框内的像素值设置为50
        # 将框的边线上的像素值设置为50
        data[x1:x2+1, y1:y1+1, z1:z2+1] = 1.1
        data[x1:x2+1, y2:y2+1, z1:z2+1] = 1.1
        data[x1:x1+1, y1:y2+1, z1:z2+1] = 1.1
        data[x2:x2+1, y1:y2+1, z1:z2+1] = 1.1
        data[x1:x2+1, y1:y2+1, z1:z1+1] = 1.1
        data[x1:x2+1, y1:y2+1, z2:z2+1] = 1.1
        # print(f'data max is {data.max()}')
    
    # 保存结果
    # affine =  np.diag([-0.7, -0.7, 0.7, 1.])
    affine = np.array([[-0.7, 0, 0, 0], [0, -0.7, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 1]])
    # resample = tio.Resample(0.7)
    # resampled_img = resample(img)
    # print(resampled_img.affine)
    
    new_img = tio.ScalarImage(tensor=torch.tensor(data).unsqueeze(0), affine=affine)
    new_img.save(output_path)
    return data

# if __name__ == '__main__':

#     import yaml
#     config_path = '/public_bme/data/xiongjl/nnDet/csv_files/config.yaml'
#     with open(config_path, 'r') as stream:
#         config = yaml.safe_load(stream)

#     evaluator = DetectionEvaluator(config)
#     txt_paths = evaluator(self.network, self.current_epoch, self._timestamp, self.fold) # generate the txt file

#     result = plot(config, txt_paths, self.current_epoch, self._timestamp, self.fold)




    