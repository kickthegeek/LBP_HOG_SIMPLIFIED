import numpy as np
import cv2
import matplotlib.pyplot as plt 
import math


img = cv2.imread("face.png")
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.resize(img , (400,400))
height, width= img.shape


# true if mouse is down 
drawing = False
ix,iy = -1,-1
bord = 64

img1 = img.copy()


diffx = np.zeros((height,width),np.int8)

for i in range(0, height):
    for j in range(0, width):
        try:
            diffx[i,j] = img[i+1,j]-img[i-1,j]
        except:
            diffx[i,j] = 0


print(diffx.shape)
print(diffx)

diffy = np.zeros((height,width),np.int8)

for i in range(0, height):
    for j in range(0, width):
        try:
            diffy[i,j] = img[i,j+1]-img[i,j-1]
        except:
            diffy[i,j] = 0


print(diffy)
print(diffy.shape)
print(diffy.min())
print(diffy.max())

# kernelx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
# diffx= cv2.filter2D(img,-1,kernelx)
# kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])
# diffy= cv2.filter2D(img,-1,kernely)
print("min diff")
print(diffy.min())
degrees = np.zeros((height,width),np.float32)

for i in range(0, height):
    for j in range(0, width):
        if diffx[i][j] != 0:
            degrees[i,j] = math.degrees(np.arctan2(diffy[i][j],diffx[i][j]))
            
print(degrees)
print(degrees.max())
print(degrees.min())
# gx = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=1)
gradient_x = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=1)
gradient_y = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=1)
gx = diffx
gy = diffy

final_img = cv2.vconcat([gradient_x, gradient_y])
# final_img = img
hist2 = degrees.ravel()
       

def mouse_pos(event,x,y,flag,param):
    
    global ix,iy,drawing,img,bord,gradient_x,gradient_y,final_img,bloc1
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        startX = int(x - bord/2)
        startY = int(y - bord/2)
        endX = int(x + bord/2)
        endY = int(y + bord/2)
        cv2.rectangle(gradient_x, (startX, startY), (endX, endY), (0, 0, 0), 2)
        cv2.rectangle(gradient_y, (startX, startY), (endX, endY), (0, 0, 0), 2)
        final_img = cv2.vconcat([gradient_x, gradient_y])
       
        
    elif event == cv2.EVENT_MOUSEMOVE:
            pass
            #print(ix,iy)
            #find_best(ix,iy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
       
        
        startX = int(x - bord/2)
        startY = int(y - bord/2)
        endX = int(x + bord/2)
        endY = int(y + bord/2)
        #f1 = img1[startX:endX,startY:endY]

        bloc_degrees = degrees[startY:endY,startX:endX]
        bloc = img[startY:endY,startX:endX]
        #calculer les histogrammes 
        hist1 = bloc_degrees.ravel()
        

        plt.figure(figsize=(20,17))
        plt.subplot(3,2,1)
        plt.title("Bloc")
        plt.imshow(bloc, cmap="gray")
        plt.subplot(3,2,2)

        plt.hist(hist1,int(8),[-180,180]);
        plt.show()

        
        #reset 
        gradient_x = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=1)
        gradient_y = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=1)
        final_img = cv2.vconcat([gradient_x, gradient_y])
        
        
    
###################
## SHOWING THE IMAGE


cv2.namedWindow(winname='result')
cv2.setMouseCallback('result',mouse_pos)


while True:

    cv2.imshow('result',final_img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()