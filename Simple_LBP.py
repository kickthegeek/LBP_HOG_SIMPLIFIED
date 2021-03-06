import numpy as np
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread("Untitled.jpg")
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img = cv2.resize(img , (800,600))
height, width= img.shape


# true if mouse is down 
drawing = False
ix,iy = -1,-1
bord = 64




def voisin(img, center, x, y): 
    val = 0
    try: 
        if img[x][y] >= center: 
            val = 1     
    except:
        pass
    return val

# Function for calculating LBP 
def lbp(img, x, y): 
   
    center = img[x][y] 
    
    matrix = [] 
    # top left 
    x1,y2=x-1,y-1
    
    matrix.append(voisin(img, center, x1, y2)) 
    # top 
    x1,y2=x-1, y
    
    matrix.append(voisin(img, center, x1, y2)) 
    # top right 
    x1,y2= x-1, y + 1
    
    matrix.append(voisin(img, center, x1, y2)) 
    # right 
    x1,y2=x, y + 1
   
    matrix.append(voisin(img, center, x1,y2)) 
    # bottom right 
    x1,y2=x + 1, y + 1
    
    matrix.append(voisin(img, center, x1, y2)) 
    # bottom 
    x1,y2=x + 1, y
   
    matrix.append(voisin(img, center, x1, y2)) 
    # bottom left 
    x1,y2=x + 1, y-1
    
    matrix.append(voisin(img, center, x1, y2)) 
    # left 
    x1,y2=x, y-1
    
    matrix.append(voisin(img, center, x1, y2)) 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(matrix)): 
        val += matrix[i] * power_val[i]
    return val



img_lbp = np.zeros((height, width),np.uint8)
for i in range(0, height): 
    for j in range(0, width): 
        img_lbp[i, j] = lbp(img, i, j)

#cv2.imwrite('result1.jpg',img_lbp)

img1 = img.copy()
img2 = img_lbp.copy()

#bloc a comparer avec 
i_startX = int(img1.shape[0]/2)
i_startY = int(img1.shape[1]/2)
i_endX = i_startX + bord
i_endY = i_startY + bord

bloc1 = img2[i_startX:i_endX,i_startY:i_endY]

cv2.rectangle(img1, (i_startX, i_startY), (i_endX, i_endY), (0, 0, 0), 2)
cv2.rectangle(img2, (i_startX, i_startY), (i_endX, i_endY), (0, 0, 0), 2)

final_img = cv2.vconcat([img1, img2])
def mouse_pos(event,x,y,flag,param):
    
    global ix,iy,drawing,img,img_lbp,bord,img1,img2,final_img,bloc1
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        startX = int(x - bord/2)
        startY = int(y - bord/2)
        endX = int(x + bord/2)
        endY = int(y + bord/2)
        cv2.rectangle(img1, (startX, startY), (endX, endY), (0, 0, 0), 2)
        cv2.rectangle(img2, (startX, startY), (endX, endY), (0, 0, 0), 2)
        final_img = cv2.vconcat([img1, img2])
       
        
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

        bloc2 = img2[startX:endX,startY:endY]
        #calculer les histogrammes 
        hist1 = bloc1.ravel()
        hist2 = bloc2.ravel()

        # plt.figure(figsize=(9, 3))
        # plt.subplot(121)
        # plt.hist(hist1, 256, [0, 256])
        # plt.subplot(122)
        # plt.hist(hist2, 256, [0, 256])
        # plt.show()

        # plt.figure(figsize=(9, 4))
        # plt.subplot(121)
        # plt.imshow(bloc1, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(bloc2, cmap='gray')
        # plt.show()
        plt.figure(figsize=(20,17))
        plt.subplot(3,2,1)
        plt.title("Bloc1")
        plt.imshow(bloc1, cmap="gray")
        plt.subplot(3,2,2)
        plt.hist(hist1.ravel(),256,[0,256]);
        plt.subplot(3,2,3)
        plt.title("Bloc2")
        plt.imshow(bloc2, cmap="gray")
        plt.subplot(3,2,4)
        plt.hist(hist2.ravel(),256,[0,256])
        plt.show()

        #calculer MSE 
        s = 0
        for i in range(len(hist1)):
            x = np.power(hist1[i]-hist2[i],2)
            s+=x
        s/=len(hist1)

        print("MSE ENTRE LES DEUX BLOQUES")
        print(s)


        #reset 
        img1 = img.copy()
        img2 = img_lbp.copy()

        #bloc a comparer avec 
        i_startX = int(img1.shape[0]/2)
        i_startY = int(img1.shape[1]/2)
        i_endX = i_startX + bord
        i_endY = i_startY + bord

        bloc1 = img2[i_startX:i_endX,i_startY:i_endY]

        cv2.rectangle(img1, (i_startX, i_startY), (i_endX, i_endY), (0, 0, 0), 2)
        cv2.rectangle(img2, (i_startX, i_startY), (i_endX, i_endY), (0, 0, 0), 2)

        final_img = cv2.vconcat([img1, img2])
        
        
    
###################
## SHOWING THE IMAGE


cv2.namedWindow(winname='result')
cv2.setMouseCallback('result',mouse_pos)


while True:

    cv2.imshow('result',final_img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()