import cv2 as cv #pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import os

from src.thresholding import Threshold_Demo

asset_path = "C:\\Users\\timoh\\OneDrive\\Dokumente\\Bildung\\Programmieren\\sudoku-solver\\assets"

def load_image(filename):
    new_path = os.path.join(asset_path, filename)
    return cv.imread(new_path,0)

def show_image(img):
    cv.imshow("Bild",img)
    cv.waitKey()

def crop_image(image, x,y):
    x1,x2 = x
    y1,y2 = y
    return image[x1:x2,y1:y2]

def resize_image(image):
    return cv.resize(image, (400,400))

def edge_detection(img, detector):
    if detector == 'sobelx':
        return cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    elif detector == 'sobely':
        return cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    elif detector == 'canny':
        return cv.Canny(img,50,150,None,3)

def write_img(file_name, img):
    new_path = os.path.join(asset_path, file_name)
    cv.imwrite(f"assets/{file_name}",img)

def center_sudoku(img):
    img = crop_image(img,(500,3500),(0,3000))
    img = resize_image(img)
    return img

def line_detection(edges, img):
    lines = cv.HoughLines(edges, 1, np.pi / 180,120,None,0,0)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b= np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1 = (int(x0 + 1000*(-b)),int(y0 + 1000*(a)))
        pt2 = (int(x0 -1000*(-b)),int(y0 -1000 *(a)))

        cv.line(img, pt1,pt2,(0,0,255),2)
    cv.imshow("Lines", img)
    cv.waitKey()

def main():
    img = load_image("upload.jpg")
    img = center_sudoku(img)

    img = Threshold_Demo(img)
    show_image(img)
    
    edges = cv.Canny(img,50,150,None, 3)
    line_detection(edges,img)
    return img
    


if __name__=="__main__":
    
    img = load_image("Sudoku_angle.jpg")
    img = center_sudoku(img)

    img = Threshold_Demo(img)
    show_image(img)
    sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=5)
    
    edges = cv.Canny(img,50,150,None, 3)
    line_detection(edges,img)
