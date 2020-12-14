import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def val_jaccard_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    jaccard_score = intersect / (ref_vol + seg_vol + 1.0) # remove 2* from denominator 
    # Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
    # Jaccard = |A∩B| / |A∪B| = TP / (TP + FP + FN)
    jaccard_score = torch.FloatTensor([i if i>0.65 else 0 for i in jaccard_score])
    jaccard_mean_score = torch.mean(jaccard_score)

    return jaccard_mean_score

def jaccard_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_score = torch.FloatTensor([i if i>0.65 else 0 for i in iou_score])
    jaccard_threshold = torch.mean(iou_score)

    return jaccard_threshold

# All codes below are Experimentals
    
def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred) # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect))/(union - intersect +  1e-7)

def compute_jaccard(y_true, y_pred):
    mean_jaccard = 0.
    thresholded_jaccard = 0.
    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard/y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard/y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard

def iou_numpy(labels, outputs):
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + 1e-06 / (union + 1e-06))
    return iou

def jaccard_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    j = (intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth)
    if (j < 0.65):
        return torch.mean(j)
    return torch.mean(j)

def jaccard_coef_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    j = -(intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth)
    if (j > 0.65):
        j = j - 1
    return j