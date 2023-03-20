import json
#from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

#import tensorflow as tf

import sudoku_cv

ASSET_DIR = "/Users/timoh/OneDrive/Dokumente/Bildung/Programmieren/old_version/assets/"

def digit_images_from_sudoku(image_path):
    image = Image.open(image_path)
    sudoku = sudoku_cv.extract_sudoku(image)

    digits = []
    for i in range(9):
        for j in range(9):
            digit_box_coordinates = (j*30,i*30,j*30+30,i*30+30)
            digits.append(sudoku.crop(digit_box_coordinates))
    
    return digits

def directional_analysis(digit_img):
    digit_img = digit_img.crop((3,3,25,25))
    np_image = sudoku_cv.pillow2np(digit_img)
    np_image = sudoku_cv.draw_image_border(np_image)
    
    start = sudoku_cv.get_point_on_contour(np_image)
    if start is None:
        return []
    x,y=start
    
    direction = 0 # 0: right, 1: right-down, 2: down ...
    directions_tracing =[]
    coordinates = []

    while True:

        x,y,direction = sudoku_cv.next_border_clockwise(np_image,x,y,direction)
        directions_tracing.append(direction)
        coordinates.append((x,y))

        if (x,y)==start:
            break
    
    return directions_tracing
def analyse_sudoku_digits(digits):
    analysis= [ ]
    for digit in digits:
        i=0
        directions = directional_analysis(digit)
        analysis.append(directions)

    with open(ASSET_DIR+"analysis.json","w+") as file:
        json.dump(analysis,file)

def digit_preprocess(digit_img):
    # crop to 28x28 pixels
    digit_img = digit_img.crop((1,1,29,29))
    digit_img = sudoku_cv.grayscale_img(digit_img)
    digit_img = sudoku_cv.adaptive_thresholding(digit_img)
    np_image= sudoku_cv.pillow2np(digit_img)
    np_image = np_image / 255.0
    np_image=np_image.astype('float32')
    
    return np_image

def is_empty(img_digit):
    img_digit = sudoku_cv.draw_image_border(img_digit,val=0,border_width=4)
    pixels_of_digit = np.count_nonzero(img_digit)
    if pixels_of_digit < 13:
        return True
    return False

def preprocess_2(img_digit):
    img_digit = sudoku_cv.draw_image_border(img_digit, val=0, border_width=3)
    return img_digit

def add_border(np_img, border_width):
    height,width =np_img.shape
    img_with_border = np.zeros((height+2*border_width,width+2*border_width))
    for i in range(height):
        for j in range(width):
            img_with_border[i+border_width,j+border_width]=np_img[i,j]
    return img_with_border
            
def preprocess_vgg(np_image):
    np_image=add_border(np_image,2)
    np_image=np.expand_dims(np_image,axis=-1)
    rgb_img = np.repeat(np_image[..., np.newaxis], 3, -1)
    rgb_img = np.resize(rgb_img,(1,32,32,3))
    return rgb_img

def make_digit_ready(digit_img):
    digit_img = digit_img.crop((1,1,29,29))
    digit_img = sudoku_cv.grayscale_img(digit_img)
    digit_img = sudoku_cv.adaptive_thresholding(digit_img)
    return digit_img


def get_prediction(img_digit):

    model = tf.keras.models.load_model(ASSET_DIR+"digit_model.h5")

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    if is_empty(img_digit): 
        return ""
    #img_digit = preprocess_2(img_digit)
    #img_digit = preprocess_vgg(img_digit)
    img_digit = np.reshape(img_digit,(1,28,28))
    prediction = probability_model.predict(img_digit,batch_size=1)
    print(prediction)
    return prediction.argmax()

def digit_recognition(images): 
    
    digits = []    
    for index,image in enumerate(images):
        print(index)
        digit = get_prediction(image)
        digits.append(digit)
    
    return digits

if __name__=="__main__":
    # get a list of pictures of the digits
    digits_img = digit_images_from_sudoku(ASSET_DIR+"Sudoku_front.jpg") 

    digit = make_digit_ready(digits_img[2])
    digit.save("Digit_Img_1.png")
    digit = make_digit_ready(digits_img[5])
    digit.save("Digit_Img_2.png")
    digit = make_digit_ready(digits_img[7])
    digit.save("Digit_Img_3.png")
    digit = make_digit_ready(digits_img[11])
    digit.save("Digit_Img_4.png")
    digit = make_digit_ready(digits_img[12])
    digit.save("Digit_Img_5.png")
    # add preprocessing to pictures
    digits_img = [digit_preprocess(digit) for digit in digits_img]
    
    # turn the images into integers
    digits = digit_recognition(digits_img)
    print(digits)
    
