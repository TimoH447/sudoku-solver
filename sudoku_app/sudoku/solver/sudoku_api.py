import numpy as np
from PIL import Image

from . import sudoku_cv
from .aws_interface import AwsInterface

def digit_images_from_sudoku(sudoku):
    digits = []
    for i in range(9):
        for j in range(9):
            digit_box_coordinates = (j*30,i*30,j*30+30,i*30+30)
            digits.append(sudoku.crop(digit_box_coordinates))
    
    return digits

def digit_preprocess(digit_img):
    # crop to 28x28 pixels
    digit_img = digit_img.crop((1,1,29,29))
    # gray scale
    digit_img = sudoku_cv.grayscale_img(digit_img)
    # thresholding
    digit_img = sudoku_cv.adaptive_thresholding(digit_img)
    
    return digit_img

def is_empty(img_digit):
    img_digit = sudoku_cv.pillow2np(img_digit)
    img_digit = sudoku_cv.draw_image_border(img_digit,val=0,border_width=4)
    pixels_of_digit = np.count_nonzero(img_digit)
    if pixels_of_digit < 13:
        return True
    return False


def get_prediction(digit_image):
    if is_empty(digit_image): 
        return ""
    aws_interface = AwsInterface(in_development= True)
    predicted_img = aws_interface.lambda_digit_recognition("sudoku-solver-bucket","images/digit.png")
    return predicted_img
    

def convert_image_to_digits(image_file):
    """
    this function takes an image file. 
    It extract the sudoku from the image and reads out the digits. 
    A list is return with the digits of the sudoku.
    """
    print("Start convert_image_to_digits")
    aws_interface = AwsInterface(in_development= True)

    img = Image.open(image_file)

    # extract the sudoku from the image and get the square of the sudoku
    print("extract sudoku")
    img = sudoku_cv.extract_sudoku(img)
    
    # get a list of images of each digit
    print("get digit image list")
    digit_images = digit_images_from_sudoku(img)
    # iterate through each cell
    digits = []
    for digit in digit_images:
        # preprocess image of the cell
        digit = digit_preprocess(digit)

        # upload the image to s3
        print("upload to s3")
        response_upload = aws_interface.upload_photo(digit,"images/digit.png")
        # ocr the image
        print("predict")
        predicted_digit = get_prediction(digit)
        digits.append(predicted_digit)

        # delete the image
        response_delete = aws_interface.delete_photo("sudoku-solver-bucket","images/digit.png")

    return digits

