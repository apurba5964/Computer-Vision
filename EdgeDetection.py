import cv2
import numpy as np


img = cv2.imread("task1.png", 0)

sobel_x = [[-1, 0, 1], [-2,0,2], [-1,0,1]]
sobel_y = [[-1, -2, -1], [0,0,0], [1,2,1]]
print(sobel_x)
print(sobel_y)


def getSobelImage(img,sobel):
    height, width= img.shape
    sobelImage=[[0 for col in range(width)] for row in range(height)]
    for x in range(1,height-1):
        for y in range(1,width-1):
            pixel_x =           (sobel[0][0] * img[x-1][y-1]) + \
                            (sobel[0][1] * img[x-1][y]) + \
                            (sobel[0][2] * img[x-1][y+1]) + \
                            (sobel[1][0] * img[x][y-1])   +\
                             (sobel[1][1] * img[x][y])   + \
                             (sobel[1][2] * img[x][y+1]) + \
                             (sobel[2][0] * img[x+1][y-1]) + \
                             (sobel[2][1] * img[x+1][y]) + \
                             (sobel[2][2] * img[x+1][y+1])
            sobelImage[x][y]=pixel_x
    return np.asarray(sobelImage)        
      
def normalizeMatrix(img):
    h,w=img.shape
    currMax=0
    for x in range(0,h):
        for y in range(0,w):
            if (img[x][y]<0):
                img[x][y]= 0-img[x][y]
            if (currMax<img[x][y]):
                currMax=img[x][y]

    for i in range(0,h):
        for j in range(0,w):
            img[i][j]=(img[i][j]/currMax)*255

    return img



sobelImageX=getSobelImage(img,sobel_x)
sobelImageY=getSobelImage(img,sobel_y)
edge_x=normalizeMatrix(sobelImageX)
edge_y=normalizeMatrix(sobelImageY)
cv2.imwrite("edge_x"+'.png',edge_x)
cv2.imwrite("edge_y"+'.png',edge_y)    
  
