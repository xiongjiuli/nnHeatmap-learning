import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#

    pred = pred.permute(0,2,3,4,1)
    target = target.permute(0,2,3,4,1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 1, 3)
    
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    # print(f'in the f1 loss the loss.max() is {loss.max()}')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def map_nonzero_to_0_max(arr):
    # 确定非零元素
    non_zero = arr[arr != 0]
    
    # 如果数组全为零，直接返回原数组
    if non_zero.size == 0:
        return arr
    
    # 计算非零元素的最大值
    max_val = non_zero.max()
    
    # 映射非零值到0-1区间（相对于最大值）
    # 注意，这里假定最大值不为0，如果最大值为0（即所有非零元素都是负数），这会导致除以零的错误
    arr[arr != 0] = arr[arr != 0] / max_val
    
    return arr

def weighted_mse_loss(y_pred, y_true, a=1, b=1000):
    # 假设y_true和y_pred已经是PyTorch的张量，并且它们都在同一个设备上（例如GPU）
    
    # 计算权重，注意这里使用的是torch.where而不是np.where
    y_true_map = map_nonzero_to_0_max(y_true)
    weights = torch.where(y_true > 0, a + b * y_true_map, torch.ones_like(y_true))
    
    # 计算加权MSE
    # 注意保持所有操作都在PyTorch张量上进行
    loss = torch.mean(weights * (y_true - y_pred) ** 2)
    
    return loss

def Mse_Loss(pred, target):
    pred = pred.float()
    target = target.float()
    mse_loss = nn.MSELoss(reduction='mean')
    loss = mse_loss(pred, target)
    return loss


def has_invalid_values(vector):
    return torch.any(torch.isinf(vector)) or torch.any(torch.isnan(vector))




def focal_loss(preds, targets, weight):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w x d)
        gt_regr (B x c x h x w x d)
    '''
    # print(f'preds is {preds}')
    preds = preds.permute(0, 2, 3, 4, 1)
    targets = targets.permute(0, 2, 3, 4, 1)

    pos_inds = targets.ge(0.9).float()
    neg_inds = targets.lt(0.9).float()
    # print(f'pos_inds is {pos_inds.max()}')
    # print(f'neg_inds is {neg_inds.max()}')
    neg_weights = torch.pow(1 - targets, weight)
    # print(f'neg_weights is {neg_weights.max()}')

    loss = 0
    # print(f'the type of preds is {type(preds)}')
    # for pred in preds:
        # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)
    # print(f'preds.min() is {preds.min()}')
    # print(f'preds.max() is {preds.max()}')
    # print(f'in the loss shape is {preds.shape}')
    # print(f'the target shape is {pos_inds.shape}')
    pos_loss = torch.log(pos_preds) * torch.pow(1 - preds, 2) * pos_inds * 100.
    # print(f'the pos_loss is {has_invalid_values(torch.log(pos_preds))}')
    # print(f'in the pos loss, the torch.log(preds).max() is {torch.log(preds).min()}, the torch.pow(1 - preds, 2).max() is {torch.pow(1 - preds, 2).max()}, the pos_inds.max() is {pos_inds.max()}')

    neg_preds = torch.clamp((1-preds), min=1e-4, max=1 - 1e-4)
    # print(f'neg_preds min is {neg_preds.min()}')
    neg_loss = torch.log(neg_preds) * torch.pow(preds, 2) * neg_weights * neg_inds
    # print(f'the torch.log(1 - preds) is {has_invalid_values(torch.log(1 - preds))}')
    # print(f'in the neg loss, the torch.log(1 - preds).max() is {torch.log(1 - preds).max()}, the torch.pow(preds, 2).max() is {torch.pow(preds, 2).max()}, the neg_weights * neg_inds.max() is {(neg_weights * neg_inds).max()}')
    # print(f'the pos loss is {pos_loss}, the neg loss is {neg_loss}')
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # print(f'in the focal loss the neg loss is {neg_loss}, and the pos_loss is {pos_loss}')
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)

def focal_loss_mse(pred, tar, a=0.0001):
    pred = torch.sigmoid(pred)
    loss_focal = focal_loss(pred, tar, 2)
    loss_mse = weighted_mse_loss(pred, tar)
    # print(f'focal loss is {loss_focal}, mse loss is {loss_mse}')
    loss = (a * loss_focal + loss_mse)
    return loss

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors) # weights is [0.53333333 0.26666667 0.13333333 0.06666667 0.        ]
        self.loss = loss

    def forward(self, output, target, whd_target):
        weights = self.weight_factors

        loss_sum = sum([weights[i] * focal_loss_mse(out[:, 0:1, :, :, :], tar) for i, (out, tar) in enumerate(zip(output, target)) if weights[i] != 0.0])

        #     np.save('')
            # mask =  np.where(whd_target[:, 0, :, :, :] != 0, 1, 0)
        mask = torch.where(whd_target[:, 0, :, :, :] != 0, torch.tensor(1).to(whd_target.device), torch.tensor(0).to(whd_target.device))
        # np.save('/public_bme/data/xiongjl/nnDet/temp/mask.npy', mask.cpu().detach().numpy())
#        if target[0].max() > 0.5:
#            output_save = [item.cpu().detach().numpy() for item in output]
#            target_save = [item.cpu().detach().numpy() for item in target]
#            for i, vector in enumerate(output_save):
#                np.save(f'/public_bme/data/xiongjl/nnDet/temp/output_{i}.npy', vector)
#            for i, vector in enumerate(target_save):
#                np.save(f'/public_bme/data/xiongjl/nnDet/temp/target_{i}.npy', vector)
#            np.save('/public_bme/data/xiongjl/nnDet/temp/whd_target.npy', whd_target.cpu().detach().numpy())
#            np.save('/public_bme/data/xiongjl/nnDet/temp/mask.npy', mask.cpu().detach().numpy())
#            print('save the loss thing down!!!!!!!!!!!!!!!!!!')
            
        loss_f1 = reg_l1_loss(output[0][:, 1:, :, :, :], whd_target, mask)
        # print(f'l1 loss is {loss_f1}')
        loss_all = loss_sum * 100 + loss_f1
        # print(f'the l1 loss is {loss_f1}, the loss sum is {loss_sum * 100}, the loss all is {loss_all}', flush=True)
        return loss_all
    
    '''    
    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors
        loss_sum = sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])

        return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])'''
