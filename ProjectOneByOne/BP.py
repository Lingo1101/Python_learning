#! /usr/bin/env python
# coding=utf-8

from sklearn.datasets import load_digits  # 数据集
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from sklearn.cross_validation import train_test_split  # 数据集分割
import numpy as np
import pylab as pl  # 数据可视化

digits = load_digits()  # 载入数据
print(digits.data.shape)  # 打印数据集大小(1797L, 64L）

pl.gray()  # 灰度化图片
pl.matshow(digits.images[0])  # 显示第1张图片，上面的数字是0
pl.show()#这个呢