import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = cv2.imread('face.png')
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.resize(img,(400,400))






# calculate gx and gy and teta
pi = 3.14
w = img.shape[1]
h = img.shape[0]
# concatenation
img = cv2.vconcat([img, img[h-1:h, 0:w]]) # +1 row at the end
img = cv2.vconcat([img[0:1, 0:w], img]) # +1 row at the start
img = cv2.hconcat([img, img[0:h+2, w-1:w]]) # +1 column at the end 
img = cv2.hconcat([img[0:h+2, 0:1], img]) # +1 column at the start 

Gx = np.zeros((h, w),np.int8) 
Gy = np.zeros((h, w),np.int8) 
teta = np.zeros((h, w),np.int8) 

for y in range(-1, h-1):
    for x in range(-1, w-1):
        mini_matrix =  img[(y)+1:(y+3)+1, (x)+1:(x+3)+1].copy()
        Gx[y+1][x+1] = mini_matrix[1][0] - mini_matrix[1][2]
        Gy[y+1][x+1] = mini_matrix[0][1] - mini_matrix[2][1]
        gx = Gx[y+1][x+1]
        gy = Gy[y+1][x+1]
        if gx == 0:
            if gy == 0:
                teta[y+1][x+1] = -1
            else :
                if gy> 0:
                    teta[y+1][x+1] = 2
                else :
                    teta[y+1][x+1] = 6
        else :
            val = gy/gx
            degree_val_corrected = np.arctan(val)*(180/pi)
            if (gx<0):
                degree_val_corrected += 180
            if (gx>0 and gy<0):
                degree_val_corrected += 360
            if(337.5<=degree_val_corrected or degree_val_corrected<22.5):
                teta[y+1][x+1] = 0
            if(22.5<=degree_val_corrected<67.5):
                teta[y+1][x+1] = 1
            if(67.5<=degree_val_corrected<112.5):
                teta[y+1][x+1] = 2
            if(112.5<=degree_val_corrected<157.5):
                teta[y+1][x+1] = 3
            if(157.5<=degree_val_corrected<202.5):
                teta[y+1][x+1] = 4
            if(202.5<=degree_val_corrected<247.5):
                teta[y+1][x+1] = 5
            if(247.5<=degree_val_corrected<292.5):
                teta[y+1][x+1] = 6
            if(292.5<=degree_val_corrected<337.5):
                teta[y+1][x+1] = 7

plt.figure(figsize=(13,9))
plt.subplot(2,1,1)
plt.title("Gx")
plt.imshow(Gx, cmap="gray")
plt.subplot(2,1,2)
plt.title("Gy")
plt.imshow(Gy, cmap="gray")