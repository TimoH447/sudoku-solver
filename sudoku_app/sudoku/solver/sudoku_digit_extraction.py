import json
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import tensorflow as tf

import sudoku_cv

ASSET_DIR = "/Users/timoh/OneDrive/Dokumente/Bildung/Programmieren/old_version/assets/"

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


def digit_images_from_sudoku(image_path):
    sudoku = sudoku_cv.extract_sudoku(image_path=image_path,saving=False)

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
    np_image = np.reshape(np_image,(1,28,28))
    
    return np_image

def get_prediction(images):
    model.fit(x_train, y_train, epochs=5)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    for index,image in enumerate(images):
        print(index)
        print(probability_model.predict(image))

if __name__=="__main__":
    digits = digit_images_from_sudoku(ASSET_DIR+"Sudoku_front.jpg") 
    digits = [digit_preprocess(digit) for digit in digits]
    #digits = np.expand_dims(digits,-1)
    get_prediction(digits)

    
