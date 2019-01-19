# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as mp
x, y = [], []
with open('../data/abnormal.txt', 'r') as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(',')]
        x.append(data[:-1])
        y.append(data[-1])
x = np.array(x)
y = np.array(y)
model_ln = lm.LinearRegression()
model_ln.fit(x, y)
pred_y_ln = model_ln.predict(x)
model_rd = lm.Ridge(300, fit_intercept=True, max_iter=10000)
model_rd.fit(x, y)
pred_y_rd = model_rd.predict(x)
# print(sm.mean_absolute_error(y, pred_y))
# print(sm.mean_squared_error(y, pred_y))
# print(sm.median_absolute_error(y, pred_y))
print(sm.r2_score(y, pred_y_rd))
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, c='dodgerblue', alpha=0.75, s=60, label='Sample')
sorted_indices = x.T[0].argsort()
mp.plot(x[sorted_indices], pred_y_ln[sorted_indices],
        c='orangered', label='Linear')
mp.plot(x[sorted_indices], pred_y_rd[sorted_indices],
        c='limegreen', label='Linear')
mp.legend()
mp.show()
