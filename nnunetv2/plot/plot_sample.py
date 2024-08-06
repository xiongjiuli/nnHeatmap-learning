###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from pathlib import Path
import torchio as tio

def xyzwhd_to_corners(box):
    """
    将以中心点坐标和尺寸表示的立方体转换为由两个对角点表示的格式。
    
    参数:
    - x, y, z: 立方体中心的x, y, z坐标
    - width, height, depth: 立方体的宽度，高度和深度
    
    返回:
    - x1, y1, z1, x2, y2, z2: 立方体的两个对角点坐标
    """
    x, y, z, width, height, depth = box
    x1 = x - width / 2
    y1 = y - height / 2
    z1 = z - depth / 2
    
    x2 = x + width / 2
    y2 = y + height / 2
    z2 = z + depth / 2
    
    return x1, y1, z1, x2, y2, z2



def get_gtboxes(filename, train):
    # print(filename)
    data_root_path = Path('/public_bme/data/xiongjl/lymph_nodes/raw_data/')
    result = []
    if train:
        folder_path = f'/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/gt/whole_train_groundtruths'
        part = 'training'
        txt_name = 'whole_train_groundtruths'
    else:
        folder_path = f'/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/gt/whole_groundtruths'
        part = 'testing'
        txt_name = 'whole_groundtruths'
    path = os.path.join(folder_path, f'{filename}.txt')
    if not os.path.exists(path):
        img_path = data_root_path.joinpath(part).joinpath(filename)
        file_name = img_path.iterdir() # 迭代器不能够去进行索引
        file_name = list(file_name)
        if len(file_name) == 0:
            print(f'the part : {part}, the name : {filename} , have no data!!!!!!!!')
        else:
            img = tio.ScalarImage(os.path.join(img_path, file_name[0]))
        # * 读取csv文件中的世界坐标/public_bme/data/xiongjl/nnDet/csv_files/CTA_thin_std_testing_lymph_refine.csv
        worldcoord = pd.read_csv(f'/public_bme/data/xiongjl/nnDet/csv_files/CTA_thin_std_{part}_lymph_refine.csv')
        raw = worldcoord[worldcoord['image_path'].str.contains(filename)]
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
        # print(img.spacing)
        spacing = (0.78125, 0.78125, 1.0)
        resample = tio.Resample((0.78125, 0.78125, 1.0))
        img = resample(img)
        shape = img.shape[1:]
        # print(f'img.orientation is {img.orientation}')
        img_coords = []
        for coord in coords:
            # print(f'the coord[0:3] is {coord[0:3]}')
            # print(f'the origin is {origin}')
            img_coord = (np.array(coord[0:3]) - np.array(origin) * np.array([-1., -1., 1.]) ) / np.array(spacing) # img.spacing
            coord[3: 6] = coord[3: 6] / np.array(spacing)
            img_coords.append([img_coord[0], img_coord[1], img_coord[2], coord[3], coord[4], coord[5]])   #! xyzwhd

        for bbox in img_coords:
            x1, y1, z1, x2, y2, z2 = xyzwhd_to_corners(bbox)
            txt_path = f'/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/gt/{txt_name}'
            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
            with open(f"{txt_path}/{filename}.txt", 'a') as f:
                f.write(f'nodule {x1} {y1} {z1} {x2} {y2} {z2} {str(shape).replace(" ", "")}\n')
    else:
        with open(path, 'r') as f:
            for line in f:
                data = line.strip().split()
                x1, y1, z1, x2, y2, z2 = map(float, data[1:7])
                result.append([x1, y1, z1, x2, y2, z2])
    return result


def get_predboxes(config, pred_file_name, filename, confidence, fold):
    result = []
    folder_path = f'/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/3d_fullres_fold_{fold}/{pred_file_name}'   
    # folder_path = f'/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/{pred_file_name}'   
    txt_path = os.path.join(folder_path, f'{filename}.txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                confi = float(data[1])
                if confi >= confidence:
                    x1, y1, z1, x2, y2, z2 = map(float, data[2:8])
                    result.append([x1, y1, z1, x2, y2, z2])
    else:
        result.append([0., 0., 0., 0., 0., 0.])
        print(f'in the {filename} no pred bbox!')
    return result


def iou(box1, box2):
    # 计算交集（overlap）的边界
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    z1_inter = max(box1[2], box2[2])
    x2_inter = min(box1[3], box2[3])
    y2_inter = min(box1[4], box2[4])
    z2_inter = min(box1[5], box2[5])
    inter_volume = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter) * max(0, z2_inter - z1_inter)
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union_volume = box1_volume + box2_volume - inter_volume
    iou = inter_volume / union_volume
    
    return iou

def get_fn(gt, pred, iou_confi): # 得到没有被预测出来的 gt
    result = []
    b = 0
    for i in range(len(gt)):
        box = gt[i]
        overlap = False
        for j in range(len(pred)):
            p_box = pred[j]
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                b += 1
                break
        if not overlap:
            result.append(box)
    # if pred == []:
        # print(f'get_fn result is {len(result)}')
    return result


def get_fp(gt, pred, iou_confi):
    # print(f'the pred is {pred}')
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
    # if pred == []:
        # print(f'result is {result}')
    return result


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names




def plot_froc(config, pred_file_name, train_data, iou_confi, size_fliter, fold, start=50, end=105, step=5, txt=False):
    # print(pred_file_name)
    pred_file_name = pred_file_name.split('/')[-2]
    if 'train' in pred_file_name:
        part = 'train'
    else:
        part = 'test'
    if 'stageone' in pred_file_name:
        stage = 'one'
    if 'stagetwo' in pred_file_name:
        stage = 'two'
    if 'merge' in pred_file_name:
        stage = 'merge'
    else:
        stage = ''
    false_positives_per_image = []
    recall = []
    precision = []
    accuracy = []
    f1 = []
    for i in range(start, end, step):
        i = round(i * 0.01, 2)
        if train_data == True:
            path = '/public_bme/data/xiongjl/nnDet/csv_files/part_testing_trainingnames.csv'# config['training_gt'] # 指定文件夹路径
        else:
            path = '/public_bme/data/xiongjl/nnDet/csv_files/part_testing_names.csv'

        # to get the names of the training or the testing
        test_names = read_names_from_csv(path)
        # print(f'the number of test names is {len(test_names)}')
        gtall_nodes = 0
        fp_nodes = 0

        FP_ls = []
        FN_ls = []
        TP_ls = []
        
        for filename in test_names:
            ground_truth_boxes = get_gtboxes(filename, train=train_data)  # 这个就是提取gt
            pred_bboxes = get_predboxes(config, pred_file_name, filename, confidence=i, fold=fold)  # 这个就是根据confi来提取大于这个confi的bbox
            # print(f'pred box number is {len(pred_bboxes)}')
            # predall_nodes += len(pred_bboxes)

            no_predbox_FN = get_fn(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi)  # 得到没有被预测出来的gt_box FN #!这个地方应该再加上iou的一些设置
            # print(f'number of no pred box is {len(no_predbox_FN)}')
            no_predbox_FN = filter_size(no_predbox_FN, size_fliter)
            # print(f'after filter, number of no pred box is {len(no_predbox_FN)}')
            FN_ls.append(len(no_predbox_FN))

            fp = get_fp(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi) # 多检测出来的结节 FP, 是从pred里面拿的
            fp = filter_size(fp, size_fliter)
            FP_ls.append(len(fp))

            tp_nodes = (len(ground_truth_boxes) - len(no_predbox_FN)) # 测出来的结节 TP
            TP_ls.append(len(ground_truth_boxes) - len(no_predbox_FN))
            # print(f'the gt number is {len(ground_truth_boxes)}, the tp number is {tp_nodes}, the fn number is {len(no_predbox_FN)} , \nthey add is {tp_nodes + len(no_predbox_FN)}^^^^^^^^^^^^^')
            fp_nodes += len(fp)

            gtall_nodes += len(ground_truth_boxes)

        # tPs = tp_nodes / gtall_nodes  # recall 
        fp_point = np.mean(FP_ls)
        fn_point = np.mean(FN_ls)
        tp_point = np.mean(TP_ls)

        accuracy.append(tp_point/(tp_point + fp_point + fn_point))
        if (tp_point + fp_point) == 0:
            precision.append(0)
        else:
            precision.append(tp_point/(tp_point + fp_point))
        recall.append(tp_point/(tp_point + fn_point))
        f1.append((2 * tp_point) / (2 * tp_point + fp_point + fn_point))
        false_positives_per_image.append(fp_point)

    with open('/public_bme/data/xiongjl/nnDet/csv_files/evaluation_out.txt', 'a') as f:
        f.write('##############################################################\n')
        f.write('** 这个就是把匹配之后的fn(没有被预测出来的gt)和fp和tp给过滤一下的\n')
        f.write(f'the names number is {len(test_names)}\n')
        f.write(f'pred_file_name is {pred_file_name}\n')
        f.write(f'size_fliter is {size_fliter}\n')
        f.write(f'start, end, step is {start, end, step}')
        f.write(f'iou_confi is {iou_confi}')
        f.write(f'train_data is {train_data}\n')
        f.write(f'false_positives_per_image is {false_positives_per_image}\n')
        f.write(f'recall is {recall}\n')
        f.write(f'precision is {precision}\n')
        f.write(f'accuracy is {accuracy}\n')
        f.write(f'f1 is {f1}\n')

    closest_index = find_closest(false_positives_per_image, 5)
    recall_at_closest_index = recall[closest_index]

    plt.plot(false_positives_per_image, recall, marker='o', label=f'recall_{part}_{iou_confi}_{size_fliter}')

    if txt:
        labels = [f'{round(i * 0.01, 2)}' for i in range(start, end, step)]
        for i, label in enumerate(labels):
            plt.annotate(label, xy=(false_positives_per_image[i], recall[i]), xytext=(false_positives_per_image[i] + 0.006, recall[i] - 0.08),
                        arrowprops=dict(facecolor='gray', edgecolor='gray', arrowstyle='-'))
    return recall, precision, recall_at_closest_index

def filter_size(boxes, size):
    new_boxes = []
    i = 0
    # print(f'the boxes is {boxes}')
    for box in boxes:
        # print([x-y for x,y in zip(box[3:], box[0:3])])
        if max([x-y for x,y in zip(box[3:], box[0:3])]) > size:
            new_boxes.append(box)
        else:
            i += 1
    # if i > 0:
        # print(f'boxes have {i} boxes <= 8')
    return new_boxes

def find_closest(lst, target):
    """
    在列表中找到离目标值最近的元素的索引。
    """
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))



def plot_ap(recall, precision):
    recall_arr = np.array(recall)
    max_precision = np.array(precision).max()
    node = len(precision)-1
    for i in range(len(precision)-1):
        if precision[i+1] - precision[i] < 0:
            precision[-1] = 1
            node = i
            break
    for j in range(node+1, len(precision)-1):
        precision[j] = max_precision
    precision_arr = np.array(precision)
    ap_corrected = np.trapz(recall_arr, precision_arr)
    return ap_corrected


def delete_image_with_name_and_best(directory, name):
    # 遍历指定目录下的所有文件
    for file in os.listdir(directory):
        # 检查文件名是否包含特定的字符串
        if name in file and 'best' in file:
            # 构建完整的文件路径
            file_path = os.path.join(directory, file)
            # 删除文件
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def plot(config, txt_paths, epoch, timesteamp, fold):
    for path in txt_paths:
        if not path.endswith('/'):
            path = os.path.join(path, '')
        if 'train' in path:
            txt_path_training = path
            # print(f'train- have')
        else:
            txt_path_testing = path

    size_fliter = 8
    plt.figure()

    recall_test_01, precision_test_01, recall_at_closest_index_test_01 = plot_froc(config, fold=fold, pred_file_name=txt_path_testing, train_data=False, iou_confi=0.01, size_fliter=size_fliter, txt=True)
    recall_test_01, precision_test_01, recall_at_closest_index_train_01 = plot_froc(config, fold=fold,pred_file_name=txt_path_training, train_data=True, iou_confi=0.01, size_fliter=size_fliter)
    # recall_test_01, precision_test_10, _ = plot_froc(config, pred_file_name=txt_path_testing, train_data=False, iou_confi=0.1, size_fliter=size_fliter)
    # recall_train_10, precision_train_10, _ = plot_froc(config, pred_file_name=txt_path_training, train_data=True, iou_confi=0.1, size_fliter=size_fliter)
    # recall_train_30, precision_train_30, _ = plot_froc(config, pred_file_name=txt_path_training, train_data=True, iou_confi=0.3, size_fliter=size_fliter)
    # recall_test_30, precision_test_30, _ = plot_froc(config, pred_file_name=txt_path_testing, train_data=False, iou_confi=0.3, size_fliter=size_fliter)
    # recall_test_50, precision_test_50, _ = plot_froc(config, pred_file_name=txt_path_testing, train_data=False, iou_confi=0.5, size_fliter=size_fliter, txt=True)
    # recall_train_50, precision_train_50, _ = plot_froc(config, pred_file_name=txt_path_training, train_data=True, iou_confi=0.5, size_fliter=size_fliter)
    plt.grid()
    plt.legend()
    plt.xlabel('False Positives Per Image')
    plt.ylabel('True Positive Fraction (recall)')
    plt.title(f'fROC Curve {config["model_name"]} {timesteamp} epoch:{epoch} size_fliter:{size_fliter}')
    # 假设你想要写入的路径和文件名
    path = f"/public_bme/data/xiongjl/nnDet/png_img/{config['model_name']}/"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"/public_bme/data/xiongjl/nnDet/png_img/{config['model_name']}/{timesteamp}-fROC-{epoch}-size_fliter{size_fliter}.png")
    plt.close()

    results = {}
    print(f'test ap is {recall_at_closest_index_test_01}, train ap is {recall_at_closest_index_train_01}')
    results['recall_test'] = recall_at_closest_index_test_01
    results['recall_train'] = recall_at_closest_index_train_01


    return results



if __name__ == '__main__':

    import yaml
    import datetime
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%H%M%S")
    config_path = '/public_bme/data/xiongjl/nnDet/csv_files/config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # evaluator = DetectionEvaluator(config)
    # txt_paths = evaluator(self.network, self.current_epoch, self._timestamp) # generate the txt file
    txt_paths = ['/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/0221005100_epoch_27_test',
                 '/public_bme/data/xiongjl/nnDet/nnunetv2/plot/bbox_txt/0221005100_epoch_27_train']
    
    result = plot(config, txt_paths, 27, timestamp)   