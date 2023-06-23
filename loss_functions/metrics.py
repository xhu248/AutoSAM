import threading
import torch
import numpy as np

# PyTroch version

SMOOTH = 1e-5


def dice_pytorch(outputs: torch.Tensor, labels: torch.Tensor, N_class):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze().float()
    labels = labels.squeeze().float()
    dice = torch.ones(N_class-1).float()
    # dice = torch.ones(N_class).float()
    ## for test
    #outputs = torch.tensor([[1,1],[3,3]]).float()
    #labels = torch.tensor([[0, 1], [2, 3]]).float()

    for iter in range(1, N_class): ## ignore the background
    # for iter in range(0, N_class):
        predict_temp = torch.eq(outputs, iter)
        label_temp = torch.eq(labels, iter)
        intersection = predict_temp & label_temp
        intersection = intersection.float().sum()
        union = (predict_temp.float().sum() + label_temp.float().sum())

        if intersection>0 and union>0:
            dice_temp = (2*intersection)/(union)
        else:
            dice_temp = 0
        #print(dice_temp)
        dice[iter-1] = dice_temp #(intersection + SMOOTH) / (union + SMOOTH)
        # dice[iter] = dice_temp
    #print(dice)

    return dice  # Or thresholded.mean()

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


################ Numpy version ################
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze()

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def dice_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze()

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    dice = (intersection + SMOOTH) / (union + SMOOTH)

    return dice  # Or thresholded.mean()


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self, mode='mean'):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        Dice = 2.0 * self.total_inter / (np.spacing(1) + self.total_union + self.total_inter)
        if mode == 'mean':
            mIoU = IoU.mean()
            Dice = Dice.mean()
            return pixAcc, mIoU, Dice
        else:
            return pixAcc, IoU, Dice

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    # predict = torch.max(output, 1)[1]
    predict = torch.argmax(output, dim=1)
    # predict = output

    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass): #只区分背景和器官: nclass = 2
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor                      #model的输出
        target: label 3D Tensor                       #label
        nclass: number of categories (int)            #只区分背景和器官: nclass = 2
    """
    predict = torch.max(output, dim=1)[1]                 #获得了预测结果
    # predict = output
    mini = 1
    maxi = nclass-1                                   #nclass = 2, maxi=1
    nbins = nclass-1                                  #nclass = 2, nbins=1

    # label is: 0, 1, 2, ..., nclass-1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')

    predict = predict * (target >= 0).astype(predict.dtype)
    intersection = predict * (predict == target)            # 得到TP和TN

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))  #统计(TP、TN)值为1的像素个数，获得TN
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))        #统计predict中值为1的像素个数，获得TN+FN
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))          #统计target中值为1的像素个数，获得TN+FP
    area_union = area_pred + area_lab - area_inter                              #area_union:TN+FN+FP
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union