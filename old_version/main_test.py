from src.image_reading import *
from src.digit_recognition import digit_ocr
img = load_image("computer_sudoku.png")
img = crop_image(img, (0,int(220/9)),(0,int(220/9)))
show_image(img)
print(digit_ocr(img))