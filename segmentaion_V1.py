# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 06:32:01 2021

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
import nibabel as nib
import seaborn as sns

#set paths
paths = glob.glob('test1.png',recursive=True)
len(paths)

sns.set()

#Read image
#orig = np.array([np.asarray(Image.open(img))for img in paths])

#gray = np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)for img in tqdm(orig)])


#Original and grayscale
orig = cv2.imread('test1.png')

gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    

# CLAHE
def clahe_enhancer(test_img):
    test_img = test_img*255
    #test_img = np.uint8(test_img)
    
    equ = cv2.equalizeHist(test_img)
    clahe = cv2.createCLAHE(clipLimit = 3.0,tileGridSize = (8,8))
    cla = clahe.apply(equ)
    return equ,cla
   
    
#Plot CLAHE  
equ,cla = clahe_enhancer(gray)   
clahe_image_flattened = cla.flatten()
gray_image_flattened = gray.flatten()
    
         
## side by side comparision


f= plt.figure()
    
ax1 = f.add_subplot(2,2, 1)
ax1.set_title("Original", fontsize=20)
plt.imshow(orig)

ax2 = f.add_subplot(2,2, 2)
ax2.set_title("Gray", fontsize=20)
plt.suptitle("Gray", fontsize=20)
plt.imshow(gray)

ax3 = f.add_subplot(2,2, 3)
ax3.set_title("Equlization", fontsize=20)
plt.suptitle("Equlization", fontsize=20)
plt.imshow(equ)

ax4 = f.add_subplot(2,2, 4)
ax4.set_title("CLAHE", fontsize=20)
plt.suptitle("CLAHE", fontsize=20)
plt.imshow(cla)

plt.show()


## side by side comparision
f = plt.figure()
ax1 = f.add_subplot(1,2, 1)
ax1.set_title("Gray_Scale", fontsize=20)
sns.distplot(gray_image_flattened)

ax2 = f.add_subplot(1,2, 2)
ax2.set_title("CLAHE", fontsize=20)
sns.distplot(clahe_image_flattened)
plt.show()



# thresh = [cv2.threshold(cla, np.mean(cla), 255, cv2.THRESH_BINARY_INV)[1] for img in tqdm(cla)]

# edges = [cv2.dilate(cv2.Canny(cla, 0, 255), None) for img in tqdm(thresh)]



# plt.figure()
# for i, edge in enumerate(edges[0:16]):
#     plt.subplot()
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))
# plt.suptitle("Edges", fontsize=20)
# plt.show()


# cnt = sorted(cv2.findContours(cla, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

# cv3 = cv2.findContours(cla, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# mask = np.zeros((256,256), np.uint8)

# cv4 =cv2.drawContours(mask, [cnt], -1, 255, -1)

# plt.imshow(cv4)

# dst = cv2.bitwise_and(gray, gray, mask=mask)

# masked = []
# segmented = []
# for i, img in tqdm(enumerate(edges)):
#     cnt = sorted(cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
#     mask = np.zeros((256,256), np.uint8)
#     masked.append(cv2.drawContours(mask, [cnt],-1, 255, -1))
#     dst = cv2.bitwise_and(orig[i], orig[i], mask=mask)
#     segmented.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))


# plt.figure()
# for i, maskimg in enumerate(masked[0:16]):
#     plt.subplot()
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(maskimg, cmap='gray')
# plt.suptitle("Mask", fontsize=20)
# plt.show()



