import numpy as np 
import cv2
import pytesseract

treshold = 70

def dr_preprocessing(img):
    img = (img > treshold).astype(np.unit8)
    return img

def digit_ocr(img):
    return pytesseract.image_to_string(img)
