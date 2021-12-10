import torch
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

import math

#----------------------------------设置auc参数设置-------------------------------------------------
my_colors = ["#1EB2A6","#ffc4a3","#e2979c","#F67575"]
fontsize = 14
#---------------------------------------设置auc参数结束--------------------------------------------
def roc_plot(label=None,scores=None,savePath=None, name=None, save=False):
    """
    inptu:label,and
    """
    fpr, tpr, threshold = roc_curve(label, scores)  ###计算真正率和假正率
    roc_auc = roc_auc_score(label, scores)  ###计算auc的值

    # plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color=my_colors[3], label='AUC: %0.4f' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color=my_colors[0],linestyle='--')

    # 设置标题
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.title(f'{name} Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save:
        plt.savefig(os.path.join(savePath, name))
    return roc_auc


def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        noisy_image = ins + noise
        if noisy_image.max().data > 1 or noisy_image.min().data < -1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
            if noisy_image.max().data > 1 or noisy_image.min().data < -1:
                raise Exception('input image with noise has values larger than 1 or smaller than -1')
        return noisy_image
    return ins