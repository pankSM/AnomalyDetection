"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import Dataset, DataLoader
# from lib.data.datasets import get_cifar_anomaly_dataset
# from lib.data.datasets import get_mnist_anomaly_dataset
# from lib.data.datasets import get_mnist_anomaly_dataset
# from lib.data.anomaly_data import AbnomalyDataset
import torch
from PIL import Image

#--------------------------------------------------未剪切的测试数据集读取开始-------------------------------
def get_subdir(datas):
    """
    data:[("./image/001.png", 0), ("./image/002.png", 1)]

    return: label
        good in name:0
        good not in name:1
    """
    # files = list()
    label_list = list()
    # image = list()
    for data in datas.imgs:
        name, label = data
        if "good" in name:
            label  = 0.
            label_list.append(label)
        else:
            label = 1.
            label_list.append(label)
    return label_list


def GetTestData(opt):

    """
    imageFolder默认图像数据目录结构
    root
    .
    ├──dog
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──cat  
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──...

    return:(image, label)
    """
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = ImageFolder(os.path.join(root, "test"),
                                       transform=transform)
    sub_files = get_subdir(test_ds)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=8, pin_memory=True)
    return test_dl, sub_files 

#--------------------------------------------------未剪切的测试数据集读取结束-------------------------------


# def get_subdir(datas):
#     files = list()
#     for data in datas.imgs:
#         name = os.path.basename(os.path.dirname(data[0]))
#         if "_" in name:
#             files.append(name.split('_')[1])
#         else:
#             files.append(name)
#     return files


"""
将测试图像分块，生成分块的图像，然后合并
"""
class get_test_dataset(Dataset):
    def __init__(self, imgs_root=None, txt_file=None, transform=None):
        super(get_test_dataset, self).__init__()
        self.transform = transform
        self.imgs_root = imgs_root
        self.txt_file = txt_file
        self.imgs_block_path = list()
        self.get_imgs_block_name()

    def get_imgs_block_name(self):
        with open(self.txt_file, "r") as f:
            readlines = f.readlines()
            for line in readlines:
                line = line.replace("\n", "")
                img_block_name = os.path.join(self.imgs_root, line)
                self.imgs_block_path.append(img_block_name)

            print(f"len_imgs_block={len(self.imgs_block_path)}")
    def __getitem__(self, index):
        img_filename = self.imgs_block_path[index]
        image = Image.open(img_filename)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.imgs_block_path)


"""
这部分下载测试数据，是因为将测试数据切成小块，最后生成的结果是需要合成整张的图片.所以输入的图片应该是一幅排序好的图片.
"""
def load_testBlock(opt):
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_ds = get_test_dataset(imgs_root=os.path.join(opt.dataroot, "images"),
                               txt_file=os.path.join(opt.dataroot, "CT_filename.txt"),
                               transform=transform)
    #print(f"len(test_ds)={len(test_ds)}")
    #print(f"len(test_ds)={test_ds[0].size()}")
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False)
    #data = iter(test_dl)
    #image = data.next()
    #print(f"test_dl ={image.size()}")
    return test_dl

def labelConvert(dataset):
    """
    :param dataset:[(file, label),(file, label)]
    使用ImageFolder时候，每个文件目录d
    :return:
    """
    # dataset = ImageFolder()
    data = dataset.imgs
    label = list()
    for i in range(len(data)):
        key, val = data[i]
        if("good" in key):
            label.append(0)
        else:
            label.append(1)
    return dataset, label


def load_test_data(opt):
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = ImageFolder(os.path.join(opt.test_root, "test"),
                                       transform=transform)

    # sub_files = get_subdir(test_ds)
    test_ds, label = labelConvert(test_ds)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)
    #for idx, (data, _) in enumerate(test_dl):
    #    print(f"data.shape={data.shape}")
    return test_dl,label

def load_data_train(opt):
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    train_ds = ImageFolder(os.path.join(opt.train_root, 'train'), transform)
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)

    return train_dl

def load_val_data(opt):
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    val_ds = ImageFolder(os.path.join(opt.dataroot, 'val'), transform)
    val_dl = DataLoader(dataset=val_ds, batch_size=opt.test_batchsize, shuffle=False, drop_last=False,num_workers=8, pin_memory=True)

    return val_dl



