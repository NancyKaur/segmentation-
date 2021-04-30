# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:28:43 2021

@author: User
"""

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from tqdm.notebook import tqdm
np.random.seed(1)

paths = glob.glob('test1.png',recursive=True)

len(paths)

orig = np.array([np.asarray(Image.open(img))for img in paths])

orig.shape

plt.figure()
for i, img in enumerate(orig[0:16]):
    plt.subplot()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
plt.suptitle("Original", fontsize=20)
plt.show()


gray = np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)for img in tqdm(orig)])




plt.figure()
for i, img in enumerate(gray[0:16]):
    plt.subplot()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.suptitle("Grayscale", fontsize=20)
plt.show()


gray[0]

thresh = [cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY_INV)[1] for img in tqdm(gray)]

edges = [cv2.dilate(cv2.Canny(img, 0, 255), None) for img in tqdm(thresh)]


plt.figure()
for i, edge in enumerate(edges[0:16]):
    plt.subplot()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))
plt.suptitle("Edges", fontsize=20)
plt.show()


## side by side comparision
f = plt.figure()
ax1 = f.add_subplot(1,2, 1)
ax1.set_title("Grayscale", fontsize=20)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
ax2 = f.add_subplot(1,2, 2)
plt.imshow(cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))
ax2.set_title("Edges", fontsize=20)
plt.suptitle("Edges Detection", fontsize=20)
plt.show()



cnt = sorted(cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros((256,256), np.uint8)

cv2.drawContours(mask, [cnt], -1, 255, -1)

dst = cv2.bitwise_and(orig, orig, mask=mask)

masked = []
segmented = []
for i, img in tqdm(enumerate(edges)):
    cnt = sorted(cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256,256), np.uint8)
    masked.append(cv2.drawContours(mask, [cnt],-1, 255, -1))
    dst = cv2.bitwise_and(orig[i], orig[i], mask=mask)
    segmented.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))


plt.figure()
for i, maskimg in enumerate(masked[0:16]):
    plt.subplot()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(maskimg, cmap='gray')
plt.suptitle("Mask", fontsize=20)
plt.show()



