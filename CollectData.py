# -*- coding: utf-8 -*-
from Module.Common import *
import shutil
import numpy as np
import cv2
import os

def convert(size, box):
    '''
    将标注的xml文件标注转换为darknet形的坐标
    '''
    dw = 1./(size[0])
    dh = 1./(size[1])
    w = box[2]
    h = box[3]
    x = box[0]+w/2
    y = box[1]+h/2
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#创建各文件夹
imagesDir = staticDir+"/face_detection/images"
labelsDir = staticDir+"/face_detection/labels"
if os.path.exists(imagesDir):
    shutil.rmtree(imagesDir)
    shutil.rmtree(labelsDir)
os.mkdir(imagesDir)
os.mkdir(labelsDir)
trainImagesDir = imagesDir+"/train/"
trainLabelsDir = labelsDir+"/train/"
os.mkdir(trainImagesDir)
os.mkdir(trainLabelsDir)

#整理标注
lines = open(staticDir+"/face_detection/train/train_bbx_gt.txt","r").readlines()
lines = [line.strip() for line in lines]
dataDict = {"srcPath":[],"dstPath":[],'labelPath':[],"content":[]}
count = 0
content = []
for line in lines:
    #图片路径
    if "/" in line:
        srcPath = staticDir+"/face_detection/train/images/"+line
        dstPath = trainImagesDir+"%d.jpg"%count
        labelPath = trainLabelsDir+"%d.txt"%count
        #shutil.copyfile(srcPath,dstPath)
        dataDict["srcPath"].append(srcPath)
        dataDict["dstPath"].append(dstPath)
        dataDict["labelPath"].append(labelPath)
        if content!=[]:
            dataDict["content"].append(content)
        count += 1
        content = []
        continue
    #个数
    if len(line)<5:
        continue
    #整理内容
    box = line.split(" ")[:4]
    content.append(box)
dataDict["content"].append(content)


#开始写入和转移
for i in range(len(dataDict["srcPath"])):
    srcPath = dataDict["srcPath"][i]
    dstPath = dataDict["dstPath"][i]
    content = dataDict["content"][i]
    labelPath = dataDict["labelPath"][i]
    if not os.path.exists(srcPath):
        continue
    if len(content)>10:
        continue
    img = cv2.imdecode(np.fromfile(srcPath, dtype=np.uint8), -1) 
    size = (img.shape[1],img.shape[0])
    boxs = ""
    for box in content:
        box = [int(box[0]),int(box[1]),int(box[2]),int(box[3]),]
        box = list(convert(size,box))
        box = ["0"]+[str(box[0]),str(box[1]),str(box[2]),str(box[3]),]
        box = " ".join(box)+"\n"
        boxs+= box
    with open(labelPath,"w") as f:
        f.write(boxs)
    img = cv2.resize(img,(640,640))
    cv2.imencode('.jpg',img)[1].tofile(dstPath)
    