
import cv2
import numpy as np
from matplotlib import pyplot as plt


posimgList=['pos_1.jpg','pos_2.jpg','pos_3.jpg','pos_4.jpg','pos_5.jpg','pos_6.jpg','pos_7.jpg','pos_8.jpg','pos_9.jpg','pos_10.jpg'
,'pos_11.jpg','pos_12.jpg','pos_13.jpg','pos_14.jpg','pos_15.jpg']

negimgList=['neg_1.jpg','neg_2.jpg','neg_3.jpg','neg_4.jpg','neg_5.jpg','neg_6.jpg','neg_8.jpg','neg_9.jpg','neg_10.jpg']


def detectCursor(imgList):
    for imgName in imgList:
    

        img_rgb = cv2.imread(imgName)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 

        #Applying Gaussian Blur
        gaussBlur = cv2.GaussianBlur(img,(3,3),0)
        #Applying Laplacian Edge detection
        laplacianImg = cv2.Laplacian(gaussBlur,cv2.CV_8U)

        template = cv2.imread('template.jpg',0)
        #Applying Laplacian Edge detection
        laplacianTemp = cv2.Laplacian(template,cv2.CV_8U)

        w, h = template.shape[::-1]

        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        for meth in methods:
            img = img_rgb.copy()
            method = eval(meth)

        # Apply template Matching
            res = cv2.matchTemplate(laplacianImg,laplacianTemp,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, we take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(img_rgb,top_left, bottom_right, 255, 2)

            cv2.imwrite(meth[4:]+'_'+imgName,img) 

detectCursor(posimgList)
detectCursor(negimgList)