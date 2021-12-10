import glob
import os
from tqdm import tqdm 
from skimage.io import imread, imsave


cropSize= (64, 64)
stepSize = 64

# 这部分剪切代码主要用于剪切MAD
def CropMAD(imageName, savePath, pattern, cropSize, stepSize, isTrain=False):
    """
    imagename:xxx/xxx/123.png
    pattern:".png" or ".jpg"
    cropSize:(crop_h, crop_w) 需要剪切的图像块的尺寸
    stepSize:每个图像块移动的步长
    isTrain:是否是用于训练
    """
    # 设置保存路径
    dstname = None
    name = os.path.basename(imageName)
    if isTrain:
        dstname = name[:-4]
    else:
        dstname = os.path.basename(os.path.dirname(name)) + "_" + name[:-4]

    # 读取图片
    image = imread(imageName)
    h, w, c = image.shape
    
    # 剪切图片基本设置
    startRow, startCol = 0, 0
    count = 0 # 计算剪切的图像块的数量

    isRowOutBoundary = False

    while(startRow + cropSize[0] <= h):

        startCol = 0
        while(startCol + cropSize[1] <= w):
            block = image[startRow : startRow + cropSize[0], startCol : startCol + cropSize[1], :]
            imsave(os.path.join(savePath, dstname +"_" +  str(count)  + pattern), block)
            startCol = startCol + stepSize
            count += 1
        
        # 如果列越界
        if(startCol + cropSize[1] > w && startCol < w):
            startCol = w - cropSize[1]
            block = image[startRow : startRow + cropSize[0], startCol : startCol + cropSize[1], :]
            imsave(os.path.join(savePath, dstname +"_" +  str(count)  + pattern), block)
            count += 1
        
        # 对行越界进行处理
        if(isRowOutBoundary == True):
            return

        if(startRow + cropSize[0] <= h) : #没有越界
            startRow = startRow + stepSize

        if(startRow + cropSize[0] > h and startRow < h) :
            startRow = h - cropSize[0]
            isRowOutBoundary = True

def CropCT():
    image_dir = "/home/smpk/project/dataset/rawCTImg/images"
    mask_dir = "/home/smpk/project/dataset/rawCTImg/masks"

def main():
    sour_path = "./data/carpet/train/good"
    dst_path = "./data/ImgBlock/carpet/train/good"
    pattern = "*.png"
    os.makedirs(dst_path, exist_ok=True)
    imageFiles = glob.glob(os.path.join(sour_path, pattern))
    for i in tqdm(range(len(imageFiles))):
        file = imageFiles[i]
        crop(file, dst_path)

if __name__ == "__main__":
    main()

