# -*- coding: utf-8 -*-
from Module.Common import *
import os,dlib,glob,numpy
from skimage import io
import pandas as pd
import numpy as np

# 人脸关键点检测器
predictor_path = staticDir+"/shape_predictor.dat"
# 人脸识别模型、提取特征值
face_rec_model_path = staticDir+"/dlib_face_recognition.dat"
# 训练图像文件夹
faces_folder_path = staticDir+'/face_identification/gallery' 

# 加载模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

candidate = []         # 存放训练集人物名字
descriptors = []       #存放训练集人物特征列表

for f in glob.glob(os.path.join(faces_folder_path,"*.jpg")):
    #print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('\\')[-1].split('.')[0])
    # 人脸检测
    dets = detector(img, 1)
    if not len(dets):
        descriptors.append(np.zeros((128,)))
        continue
    for k, d in enumerate(dets): 
        shape = sp(img, d)
        # 提取特征
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor) 
        descriptors.append(v)
        break

print('识别训练完毕！')

def getSame(path):
    img = io.imread(path)
    dets = detector(img, 1)    
    dist = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor) 
        for i in descriptors:                #计算距离
            dist_ = numpy.linalg.norm(i-d_test)
            dist.append(dist_)
    # 训练集人物和距离组成一个字典
    c_d = dict(zip(candidate,dist))                
    cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
    try:
        return cd_sorted[0][0]
        
    except:
        return "18"

if __name__ == "__main__":
#    path = r"D:\Desktop\FaceDetect\Static\face_identification\gallery\24.jpg"
#    person = getSame(path)
    paths = [staticDir+"/face_identification/probe_test/"+"00%d.jpg"%i for i in range(2475,4950)]
    data = pd.DataFrame({"path":paths})
    data["person"] = data["path"].apply(getSame)
    #data.to_excel("compare.xlsx",index=None)
    content = ""
    for i in range(data.shape[0]):
        row = data.iloc[i,:]
        path = row["path"]
        person = row["person"]
        path = path.split("/")[-1]
        line = path+" "+str(person)+"\n"
        content+=line
    with open("pred_test.txt","w") as f:
        f.write(content)