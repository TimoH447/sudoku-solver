import os

from PIL import Image, ImageFilter,ImageDraw
import numpy as np
from matplotlib import pyplot as plt

ASSET_DIR = "/Users/timoh/OneDrive/Dokumente/Bildung/Programmieren/old_version/assets/"
DEBUG = True

def rescale_img(image,DEBUG = False):
    height, width = image.height, image.width
    scaled  = image.resize((int(width/8),int(height/8)),Image.Resampling.LANCZOS)
    if DEBUG:
        scaled.show()
    return scaled

def gaussian_blur(image, radius = 7, DEBUG=False):
    result = image.filter(ImageFilter.GaussianBlur(radius))
    if DEBUG:
        result.show()
    return result

def bradley_binary_thresh(input_img):
    print(input_img.shape)
    h, w = input_img.shape

    S = w/8
    s2 = S/2
    T = 15.0

    #integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = input_img[0:row,0:col].sum()

    #output img
    out_img = np.zeros_like(input_img)    

    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = max(int(row-s2), 0)
            y1 = min(int(row+s2), h-1)
            x0 = max(int(col-s2), 0)
            x1 = min(int(col+s2), w-1)

            count = (y1-y0)*(x1-x0)

            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]

            if input_img[row, col]*count < sum_*(100.-T)/100.:
                out_img[row,col] = 255
            else:
                out_img[row,col] = 0

    return out_img

def grayscale_img(image,DEBUG=False):
    """ returns the image graysceled"""
    grayscaled = image.convert('L')
    if DEBUG:
        grayscaled.show()
    return grayscaled

def np2pillow(np_img):
    pillow_img = Image.fromarray(np_img)
    return pillow_img

def pillow2np(pillow_img):
    np_img=np.array(pillow_img)
    return np_img

def adaptive_thresholding(image,DEBUG=False):
    np_img = pillow2np(image)
    np_img = bradley_binary_thresh(np_img)
    result = np2pillow(np_img)
    if DEBUG:
        print("result thresholding")
        result.show()
    return result

def border_tracing(image):
    NBD = 1
    f=image
    for i in range(image):
        LNBD = 1 
        for j in range(image[0]):
            if f[i,j]==1 and f[i,j-1]:
                NBD +=1
                i2,j2=(i,j-1)
            elif f[i,j]>=1 and f[i,j+1]==0:
                NBD+=1
                i2,j2=(i,j+1)
                if f[i,j]>1:
                    LNBD = f[i,j] 
            else:
                #goto 4
                pass

def pre_tracing(image):
    image= np.where(image==255,1,0)
    return image

pixel_left = {
    0: (-1,0),
    1: (-1,1),
    2: (0,1),
    3: (1,1),
    4: (1,0),
    5: (1,-1),
    6: (0,-1),
    7: (-1,-1)
}
def next_border_clockwise(image,x,y,direction):
    """finds the next border element of the current contour it is following"""
    for d in range(8):
        i,j = pixel_left[direction]
        if image[x+i,y+j]!=0:
            new_direction = (direction + 6)%8 
            return x+i,y+j,new_direction

        direction += 1
        direction = direction % 8 
    return x,y,direction

def trace_border(image,start,NBD):
    contour = []

    # initialize variables
    x, y = start
    direction = 0  # 0: right, 1: right-down, 2: down, 3: down-left, 4: left, 5: left-up, 6: up, 7: up-right

    # iterate until the starting point is reached again
    while True:
        image[x,y]=NBD
        if (x,y) not in contour:
            contour.append((x, y))

        x,y,direction = next_border_clockwise(image,x,y,direction)

        if (x,y)==start:
            break

    return contour

def draw_image_border(image, val=0, border_width=1):
    height,width = image.shape
    for i in range(width):
        for k in range(border_width):
            image[k,i]=val
            image[-(k+1),i]=val
    for j in range(height):
        for k in range(border_width):
            image[j,k]=val
            image[j,-(k+1)]=val
    return image

def find_contours(image):
    height,width = image.shape
    f = image

    all_contours = []
    NBD = 2

    for row in range(1,height-1):

        for col in range(1,width-1):

            if f[row,col-1]==0 and f[row,col]==1:
                contour = trace_border(image,(row,col),NBD)
                all_contours.append(contour)
                NBD+=1

    return all_contours

def find_biggest_contour(all_contours):
    sorted_contours = sorted(all_contours,key=len)
    return sorted_contours[-1]
            
def get_point_on_contour(image):
    height,width=image.shape
    for i in range(1,height-1):
        for j in range(1,width-1):
            if image[i,j]==255:
                return i,j

def find_vertices_sudoku(np_image):
    start = get_point_on_contour(np_image)
    x,y=start
    direction = 0 # 0: right, 1: right-down, 2: down ...
    directions_tracing =[]
    coordinates = []

    while True:

        x,y,direction = next_border_clockwise(np_image,x,y,direction)
        directions_tracing.append(direction)
        coordinates.append((x,y))

        if (x,y)==start:
            break
    
    contour = Contour(directions_tracing,coordinates)
    vertices = contour.get_vertices()
    return vertices

def draw_circles_around_vertices(img, vertices, radius=5):
    
    # Create a draw object
    draw = ImageDraw.Draw(img)
    
    # Draw circles around the vertices
    for vertex in vertices:
        print(vertex)
        draw.ellipse([vertex[1]-radius, vertex[0]-radius, vertex[1]+radius, vertex[0]+radius], fill=None, outline='red')
    
    return img

def contour_tracing(image,DEBUG=False):
    image = pillow2np(image)
    pre_img = draw_image_border(image)
    pre_img = np.where(pre_img==255,1,0)

    all_contours = find_contours(pre_img) 
    biggest_contour = find_biggest_contour(all_contours) 

    iso = np.zeros_like(pre_img)
    for i,j in biggest_contour:
        iso[i,j]=255
    
    vertices = find_vertices_sudoku(iso)

    result = np2pillow(iso)
    result.convert("RGB")
    result = draw_circles_around_vertices(result,vertices)
    if DEBUG:
        result.show()
    return result

def get_arith(lst):
    m,M = min(lst),max(lst)
    if m==0 and M>=5:
        for i in range(len(lst)):
            if lst[i]<4:
                lst[i]+=8
    return np.mean(lst)

def is_vertice(prev,next):
    prev_mid = get_arith(prev)
    next_mid = get_arith(next)
    if abs(prev_mid-next_mid)>4:
        if prev_mid<4:
            prev_mid+=8
        else:
            next_mid+=8
    if abs(next_mid-prev_mid)>1.5:
        return True
    else:
        return False

def get_sudoku_contour(image,DEBUG=False):
    image = pillow2np(image)
    pre_img = draw_image_border(image)
    pre_img = np.where(pre_img==255,1,0)

    all_contours = find_contours(pre_img) 
    biggest_contour = find_biggest_contour(all_contours) 

    if DEBUG:
        iso = np.zeros_like(pre_img)
        for i,j in biggest_contour:
            iso[i,j]=255
        contour_image = np2pillow(iso)
        contour_image.show()
    return biggest_contour

class Contour:
    def __init__(self,directions, points_in_image):
        self.direction_list = directions
        self.cooredinates = points_in_image
    def get_next_dir(self,i,amount):
        next = []
        if i+amount>=len(self.direction_list):
            for j in range(i,len(self.direction_list)):
                amount-=1
                next.append(self.direction_list[j])
            for j in range(amount):
                next.append(self.direction_list[j])
        else:
            for j in range(amount):
                next.append(self.direction_list[i+j+1])
        return next
    def get_prev_dir(self,i,amount):
        prev= []
        if i-amount<=0:
            for j in range(abs(i-amount))[::-1]:
                prev.append(self.direction_list[len(self.direction_list)-j-1])
            for j in range(i):
                prev.append(self.direction_list[j])
        else:
            for j in range(i-amount,i):
                prev.append(self.direction_list[j])
        return prev
    
    def get_vertices(self):
        vertices = []
        current_vertex = []
        for i in range(len(self.direction_list)):
            if is_vertice(self.get_prev_dir(i,5),self.get_next_dir(i,5)):
                print("found vertex",i)
                current_vertex.append(i)
            elif len(current_vertex)>0:
                print("add vertex")
                middle_of_vertex_points = len(current_vertex)//2
                vertices.append(current_vertex[middle_of_vertex_points])
                current_vertex = []
        coordinate_of_vertices =[]
        for vertex in vertices:
            coordinate_of_vertices.append(self.cooredinates[vertex])
        
        return coordinate_of_vertices
    
def get_sudoku_vertices(image,contour):
    image = pillow2np(image)
    contour_image = np.zeros_like(image)
    for i,j in contour:
        contour_image[i,j]=255
    
    vertices = find_vertices_sudoku(contour_image)
    return vertices

# from so: https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def transform_img(image,vertices,vertices_transformed):
    coeffs = find_coeffs(vertices_transformed,vertices)            
    return image.transform((270,270),Image.Transform.PERSPECTIVE,coeffs,Image.Resampling.BICUBIC)

def coordinates_switched(lst):
    switched = []
    for i,j in lst:
        switched.append((j,i)) 
    return switched

def add_suffix(filepath):
    dirname, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    new_filename = filename + '_transformed' + ext
    return os.path.join(dirname, new_filename)


def extract_sudoku(image_path,saving=True):
    unmodified_image = Image.open(image_path)
    image = grayscale_img(unmodified_image)
    image = gaussian_blur(image)
    image = rescale_img(image)
    image = adaptive_thresholding(image)

    sudoku_contour = get_sudoku_contour(image)
    vertices = get_sudoku_vertices(image,sudoku_contour)
    vertices_in_transformed = [(270,270),(270,0),(0,0),(0,270)]

    transformed = transform_img(rescale_img(unmodified_image),coordinates_switched(vertices),coordinates_switched(vertices_in_transformed))
    
    if saving:
        new_filename = add_suffix(image_path)
        transformed.save(new_filename)

    return transformed



if __name__=="__main__":
    img_name = "Sudoku_front.jpg"
    color_img = Image.open(ASSET_DIR + img_name)
    img = grayscale_img(color_img)
    img = gaussian_blur(img)
    rescaled = rescale_img(img)
    rescaled.save(ASSET_DIR+"Sudoku_front_rescaled.png")
    img = adaptive_thresholding(rescaled)
    img.save(ASSET_DIR+"Sudoku_front_thresh.png")

    sudoku_contour = get_sudoku_contour(img)
    vertices = get_sudoku_vertices(img,sudoku_contour)
    vertices_in_transformed = [(270,270),(270,0),(0,0),(0,270)]

    
    rescaled = rescaled.convert("RGB")
    img = draw_circles_around_vertices(rescaled,vertices)
    img.show()
    img.save(ASSET_DIR+"Sudoku_front_vertices.png")
    image_highlight_sudoku = rescale_img(color_img)
    draw = ImageDraw.Draw(image_highlight_sudoku)
    contour_width=2
    draw.polygon(coordinates_switched(sudoku_contour),outline=(220,20,60),width=contour_width)
    image_highlight_sudoku.show()
    image_highlight_sudoku.save(ASSET_DIR + "Sudoku_front_contour.png")

    transformed = transform_img(rescale_img(color_img),coordinates_switched(vertices),coordinates_switched(vertices_in_transformed))
    transformed.show()
    transformed.save(ASSET_DIR +"Sudoku_front_transformed.png")


    