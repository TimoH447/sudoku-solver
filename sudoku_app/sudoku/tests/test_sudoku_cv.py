import numpy as np
import unittest
try:
    import context
except ModuleNotFoundError:
    import tests.context    

from solver import sudoku_cv

pixels_o = np.array([[0,0,0,0,0],
                        [0,1,1,1,0],
                        [0,1,0,1,0],
                        [0,1,1,1,0],
                        [0,0,0,0,0]])

pixels_gamma = np.array([[0,0,0,0,0],
                        [0,1,1,1,0],
                        [0,1,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,0,0]])

pixels_eck =np.array([[0,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0],
                        [0,0,1,1,0],
                        [0,0,0,0,0]])

pixels_big = np.array([
    [0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,0],
    [0,1,1,0,1,0,1,0],
    [0,1,1,0,1,0,1,0],
    [0,1,1,0,1,0,1,0],
    [0,1,1,0,1,0,1,0],
    [0,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0],
                      ])

class test_next_border(unittest.TestCase):
    def test_next_border(self):
        result = sudoku_cv.next_border_clockwise(pixels_gamma,3,1,2)
        expected_result = (2,1,6)
        self.assertEqual(result,expected_result)
    
    def test_trace_border(self):
        result = sudoku_cv.trace_border(pixels_gamma,(1,1),2)
        expected_contour = [(1,1),(1,2),(1,3),(2,1),(3,1)]
        self.assertEqual(result,expected_contour)

    def test_find_contours(self):
        pass


if __name__=="__main__":
    #unittest.main()
    result = sudoku_cv.find_contours(pixels_big)
    biggest_contour = sudoku_cv.find_biggest_contour(result)

    iso = np.zeros_like(pixels_big)
    for i,j in biggest_contour:
        iso[i,j]=1
    print(iso)