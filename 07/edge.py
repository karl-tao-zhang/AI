# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import cv2 as cv
original = cv.imread('../data/chair.jpg', cv.IMREAD_GRAYSCALE)
# print(original.shape, original.dtype)
cv.imshow('Original', original)
hor_sobel = cv.Sobel(original, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('Horizontal Sobel', hor_sobel)
ver_sobel = cv.Sobel(original, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('Verizontal Sobel', ver_sobel)
hav_sobel = cv.Sobel(original, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('Horizontal and Verizontal Sobel', hav_sobel)
laplacian = cv.Laplacian(original, cv.CV_64F)
cv.imshow('Laplacian ', laplacian)
canny = cv.Canny(original, 50, 240)
cv.imshow('Canny', canny)

cv.waitKey()
