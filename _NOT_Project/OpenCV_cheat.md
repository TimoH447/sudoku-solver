# **OpenCV Cheat sheet**
import cv2 as cv
## **Input and Output**
Loading an image from the same folder:
```python 
 img = cv.imread(filename) 
``` 
jpg files get by default 3 color channels, 
add 0 for grey scale or cv.IMREAD_GRAYSCALE as argument

Saving an image: 
```python
cv.imwrite(filename, img)
```

## **Basic Operations**
Access pixel intesity: 
If image is gray scaled then thats just the pixel value, i.e.
```python
intesity = img[y,x]
```


