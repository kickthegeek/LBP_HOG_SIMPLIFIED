import numpy as np
import cv2
import matplotlib.pyplot as plt 
import math

img = cv2.imread("face.png")
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img = cv2.resize(img , (800,600))
height, width= img.shape


# true if mouse is down 
drawing = False
ix,iy = -1,-1
bord = 64



#cv2.imwrite('result1.jpg',img_lbp)


#hog function
def hog_calc(img):
    height, width = img.shape

    image_angles = np.zeros((height, width), np.float32)
    image_angles_degrees = np.zeros((height, width), np.float32)
    #image_gx = get_gx(img)
    #image_gy = get_gy(img)

    image_gx = np.zeros((height, width),np.int8)
    image_gy = np.zeros((height, width),np.int8) 

    #fill image_gx 
    for i in range(width):
        for j in range(height):
            if i-1 >= 0 and i+1 < width:
                image_gx[i][j] = img[i+1][j] - img[i-1][j]
            else: 
                image_gx[i][j] = 0 
    #fill image_gy 
    for i in range(width):
        for j in range(height):
            if j-1 >= 0 and j+1 < height:
                image_gy[i][j] = img[i][j+1] - img[i][j-1]
            else: 
                image_gy[i][j] = 0 


    for i in range(0, height):
        for j in range(0, width):
            #if image_gx[i][j] != 0 and image_gy[i][j] != 0:
            degree = math.atan2(image_gy[i][j],image_gx[i][j])
            image_angles[i][j] = degree
            image_angles_degrees[i][j] = math.degrees(degree)
            
                

    print(image_angles)
    # plt.subplots(2,1)
    # plt.imshow(image_gx, cmap='gray')
    # plt.subplots(2,2)

    # plt.imshow(image_gy, cmap='gray')
    # plt.show()

    plt.figure(figsize=(10,9))
    plt.subplot(3,2,1)
    plt.title("gx")
    plt.imshow(image_gx, cmap="gray")
    plt.subplot(3,2,2)
    plt.title("histograme des angles")
    plt.hist(image_angles.ravel(),8,[-math.pi,math.pi])
    plt.subplot(3,2,3)
    plt.title("gy")
    plt.imshow(image_gy, cmap="gray")
    plt.subplot(3,2,4)
    plt.title("histogramme des angels deg")
    plt.hist(image_angles_degrees.ravel(),8,[-180,180])
    plt.show()

img1 = img.copy()


final_img = img1
def mouse_pos(event,x,y,flag,param):
    
    global ix,iy,drawing,img,bord,img1,final_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        startX = int(x - bord/2)
        startY = int(y - bord/2)
        endX = int(x + bord/2)
        endY = int(y + bord/2)
        cv2.rectangle(img1,  (endX, endY),(startX, startY), (0, 0, 0), 2)
        final_img = img1
       
        
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
        f1 = img[startY:endY,startX:endX]
        hog_calc(f1)
        
    
        img1 = img.copy()
        final_img = img1
        
        
    
###################
## SHOWING THE IMAGE


cv2.namedWindow(winname='result')
cv2.setMouseCallback('result',mouse_pos)


while True:

    cv2.imshow('result',final_img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()