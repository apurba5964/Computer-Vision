from math import exp
import math
import cv2
import numpy as np

def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )

def createKernel(sigma):
  #kernel_height, kernel_width = 7, 7
  kernel_radius = 3 # for an 7x7 filter
  

  # compute the actual kernel elements
  hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
  vkernel = [x for x in hkernel]
  kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]

  # normalize the kernel elements
  kernelsum = sum([sum(row) for row in kernel2d])
  kernel2d = [[x/kernelsum for x in row] for row in kernel2d]
  return kernel2d


img = cv2.imread("task2.jpg", 0)
img_rgb = cv2.imread("task2.jpg")

height, width= img.shape
#sqrt(2)=1.414
sigmaList0=[1/1.414,1,1.414,2,2*1.414]
sigmaList2=[1.414,2,2*1.414,4,4*1.414]
sigmaList4=[2*1.414,4,4*1.414,8,8*1.414]
sigmaList8=[4*1.414,8,8*1.414,16,16*1.414]



def generateOctave(img):
    height,width=img.shape
    
    op=[[0 for col in range(int(width/2))] for row in range(int(height/2))]
    x=0
    for i in range(0,height):
        y=0
        if i%2 ==0:
            continue

        for j in range(0,width):
            if j%2 ==0:
                continue
            op[x][y]=img[i][j]
            y=y+1
        x=x+1
    return np.asarray(op)          

def createImage(kernel2d,sig,img,name):
    height,width=img.shape
    img2dhororg=[[0 for col in range(width)] for row in range(height)]
    pixel_x=0
    for x in range(3,height-3):
        for y in range(3,width-3):
            for i in range(0,7):
                for j in range(0,7):
                    pixel_x=pixel_x+kernel2d[i][j]*img[x-3+i][y-3+j]
                                    
        
            img2dhororg[x][y]=pixel_x
            pixel_x=0


    cv2.imwrite(name+'_'+str(sig)+'.png',np.asarray(img2dhororg))                

octImage2=generateOctave(img)
octImage4=generateOctave(octImage2)
octImage8=generateOctave(octImage4)

for sig in sigmaList0:
    
    kernel=createKernel(sig)
    createImage(kernel,sig,img,"octave0")
for sig in sigmaList2:
    kernel=createKernel(sig)
    createImage(kernel,sig,octImage2,"octave2")
for sig in sigmaList4:
    kernel=createKernel(sig)    
    createImage(kernel,sig,octImage4,"octave4")
for sig in sigmaList8:
    kernel=createKernel(sig)    
    createImage(kernel,sig,octImage8,"octave8")

#Generate Difference of Gaussian and Keypoints Image

def getMax(x,y,dog1,dog2,dog3):
    maxInt=0
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            currMax=0
            currMax=max(dog1[i][j],dog2[i][j],dog3[i][j])
            if(maxInt<currMax):
                maxInt=currMax
            
    return maxInt 

def getMin(x,y,dog1,dog2,dog3):
    minInt=0
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            currMin=0
            currMin=min(dog1[i][j],dog2[i][j],dog3[i][j])
            if(minInt>currMin):
                minInt=currMin
            
    return minInt 


def generateKeypoints(dog1,dog2,dog3,dog4,img,factor):

    output=img
    h1,w1=dog1.shape
    for x in range(1,h1-1):
        for y in range(1,w1-1):
            if(dog2[x][y] == getMax(x,y,dog1,dog2,dog3) or dog2[x][y] == getMin(x,y,dog1,dog2,dog3)):
            #setting the threshold to 1 to descard low intensity keypoints 
                if (dog2[x][y]>1):
                    output[factor*x][factor*y]=255
            if(dog3[x][y] == getMax(x,y,dog2,dog3,dog4) or dog3[x][y] == getMin(x,y,dog2,dog3,dog4)):
            #setting the threshold to 1 to descard low intensity keypoints 
                if (dog3[x][y]>1):
                    output[factor*x][factor*y]=255      
                
    return output

def generateDOG(name,sigmaList):
    

    
     
    img0=cv2.imread(name+'_'+str(sigmaList[0])+'.png', 0)
    img1=cv2.imread(name+'_'+str(sigmaList[1])+'.png', 0)
    img2=cv2.imread(name+'_'+str(sigmaList[2])+'.png', 0)
    img3=cv2.imread(name+'_'+str(sigmaList[3])+'.png', 0)
    img4=cv2.imread(name+'_'+str(sigmaList[4])+'.png', 0)
    

    height,width=img0.shape

    dog1=[[0 for col in range(width)] for row in range(height)]
    dog2=[[0 for col in range(width)] for row in range(height)]
    dog3=[[0 for col in range(width)] for row in range(height)]
    dog4=[[0 for col in range(width)] for row in range(height)]


    for x in range(0,height):
        for y in range(0,width):
            dog1[x][y]=int(img1[x][y])-int(img0[x][y])
            dog2[x][y]=int(img2[x][y])-int(img1[x][y])
            dog3[x][y]=int(img3[x][y])-int(img2[x][y])
            dog4[x][y]=int(img4[x][y])-int(img3[x][y])
    
    
    cv2.imwrite(name+"_DOG_10"+'.png',np.asarray(dog1))
    cv2.imwrite(name+"_DOG_21"+'.png',np.asarray(dog2))
    cv2.imwrite(name+"_DOG_32"+'.png',np.asarray(dog3))
    cv2.imwrite(name+"_DOG_43"+'.png',np.asarray(dog4))
    return[np.asarray(dog1),np.asarray(dog2),np.asarray(dog3),np.asarray(dog4)]



doglist1=generateDOG("octave0",sigmaList0)

doglist2=generateDOG("octave2",sigmaList2)

doglist3=generateDOG("octave4",sigmaList4) 

doglist4=generateDOG("octave8",sigmaList8)


print(np.min(doglist1[0]))
op1=generateKeypoints(doglist1[0],doglist1[1],doglist1[2],doglist1[3],img_rgb,1)
op2=generateKeypoints(doglist2[0],doglist2[1],doglist2[2],doglist2[3],op1,2)
op3=generateKeypoints(doglist3[0],doglist3[1],doglist3[2],doglist3[3],op2,4)
op4=generateKeypoints(doglist4[0],doglist4[1],doglist4[2],doglist4[3],op3,8)

cv2.imwrite("KEYPOINTS"+'.png',op4)

