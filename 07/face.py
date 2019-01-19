# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import numpy as np
import cv2 as cv
import sklearn.preprocessing as sp
fd = cv.CascadeClassifier(
    '../data/haar/face.xml')


def search_faces(directory):
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        raise IOError("The directory '" +
                      "' desn't exist!")
    faces = {}
    for curdir, subdirs, files in \
            os.walk(directory):
        for jpeg in (file for file
                     in files if file.endswith('.jpg')):
            path = os.path.join(curdir, jpeg)
            label = path.split(os.path.sep)[-2]
            if label not in faces:
                faces[label] = []
            faces[label].append(path)
    return faces
train_faces = search_faces(
    '../data/faces/training')
codec = sp.LabelEncoder()
codec.fit(list(train_faces.keys()))
train_x, train_y = [], []
for label, filenames in train_faces.items():
    for filename in filenames:
        image = cv.imread(filename)
        gray = cv.cvtColor(image,
                           cv.COLOR_BGR2GRAY)
        faces = fd.detectMultiScale(gray, 1.1, 2,
                                    minSize=(100, 200))
        for l, t, w, h in faces:
            train_x.append(gray[t:t + h, l:l + w])
            train_y.append(int(codec.transform(
                [label])[0]))
train_y = np.array(train_y)

# 局部二值模式
# 这个就不用提取特征值了
model = cv.face.LBPHFaceRecognizer_create()
model.train(train_x, train_y)

# 测试集
test_faces = search_faces(
    '../data/faces/testing')
test_x, test_y, test_z = [], [], []
for label, filenames in test_faces.items():
    for filename in filenames:
        image = cv.imread(filename)
        gray = cv.cvtColor(image,
                           cv.COLOR_BGR2GRAY)
        faces = fd.detectMultiScale(gray, 1.1, 2,
                                    minSize=(100, 200))
        for l, t, w, h in faces:
            test_x.append(gray[t:t + h, l:l + w])
            test_y.append(int(codec.transform(
                [label])[0]))
            a, b = int(w / 2), int(h / 2)
            # 0,0,360  左上角画360度圈
            cv.ellipse(image, (l + a, t + b),
                       (a, b), 0, 0, 360,
                       (255, 0, 255), 2)
            test_z.append(image)
test_y = np.array(test_y)
pred_test_y = []
for face in test_x:
    pred_code = model.predict(face)[0]
    pred_test_y.append(pred_code)
# 退出标识
escape = False
# 可视化
while not escape:
    for code, pred_code, image in zip(
            test_y, pred_test_y, test_z):
        label, pred_label = \
            codec.inverse_transform(
                [code, pred_code])
        # 实际名字==预测名字?
        text = '{} {} {}'.format(label,
                                 '==' if code == pred_code else '!=',
                                 pred_label)
        # 放到实时图片上
        cv.putText(image, text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 2,
                   (255, 255, 255), 6)
        cv.imshow('Recognizing Face...', image)
        if cv.waitKey(1000) == 27:
            escape = True
            break
