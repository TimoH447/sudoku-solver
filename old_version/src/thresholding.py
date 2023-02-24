import matplotlib.pyplot as plt 
import cv2 as cv

def Threshold_Demo(img):
    """
    input a greyscaled image
    """
    img = img.copy()
    
    thresh = img.mean()-30
    y,x = img.shape   
    for i in range(y):
        for j in range(x):
            if img[i,j]>thresh:
                img[i,j]= 255
            else:
                img[i,j]=0
    return img