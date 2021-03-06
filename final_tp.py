import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = cv2.imread('face.png')
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.resize(img,(400,400))


def get_gy(image):
    kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    image = cv2.filter2D(image,-1,kernel)
    return image

def get_gx(image):
    kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    image = cv2.filter2D(image,-1,kernel)
    return image

height, width = img.shape

image_angles = np.zeros((height, width), np.float32)
#image_gx = get_gx(img)
#image_gy = get_gy(img)

image_gx = np.zeros((height, width),np.int8)
image_gy = np.zeros((height, width),np.int8) 

#fill image_gx 
for i in range(width):
    for j in range(height):
        if i != 0 and i!= width-1:
            image_gx[i][j] = img[i+1][j] - img[i-1][j]
        else: 
            image_gx[i][j] = 0 
#fill image_gy 
for i in range(width):
    for j in range(height):
        if j != 0 and j!= height-1:
            image_gy[i][j] = img[i][j+1] - img[i][j-1]
        else: 
            image_gy[i][j] = 0 


for i in range(0, height):
    for j in range(0, width):
        if image_gx[i][j] != 0 and image_gy[i][j] != 0:
            degree = np.arctan(image_gy[i][j]/image_gx[i][j])
            val = math.degrees(degree)
            if val == 0:
                if image_gx[i][j] < 0:
                    image_angles[i][j] = 180
                else:
                    image_angles[i][j] = 0        
            else:
                image_angles[i][j] = val

        elif image_gx[i][j] == 0 and image_gy[i][j] != 0 :
            if image_gy[i][j] > 0:
                image_angles[i][j] = 90.0
            else:
                image_angles[i][j] = 270.0
            

print(image_angles)
# plt.subplots(2,1)
# plt.imshow(image_gx, cmap='gray')
# plt.subplots(2,2)

# plt.imshow(image_gy, cmap='gray')
# plt.show()

plt.figure(figsize=(10,8))
plt.subplot(3,2,1)
plt.title("gx")
plt.imshow(image_gx, cmap="gray")
plt.subplot(3,2,2)
plt.title("histograme de l'image")
plt.hist(image_angles.ravel(),8,[-180,180]);
plt.show()