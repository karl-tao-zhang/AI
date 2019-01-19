# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import cv2 as cv
original = cv.imread('../data/table.jpg')
cv.imshow('Original', original)
gray = cv.cvtColor(original,
                   cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
mixture = original.copy()
cv.drawKeypoints(original, keypoints, mixture,
                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Mixture', mixture)
cv.waitKey()
