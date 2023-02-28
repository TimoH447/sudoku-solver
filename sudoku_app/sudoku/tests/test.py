import unittest
try:
    import context
except ModuleNotFoundError:
    import tests.context    

from solver import solver

# sudokus
empty_sudoku = ["","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","",""]

mid_sudoku = [1,"",8,"","",5,7,"","",
                "",6,"",9,"","","",2,8,
                "",9,"",2,"",7,6,"","",
                "","","","",7,3,4,"","",
                "",3,7,"","","","",5,1,
                4,"","",6,"",9,"",3,"",
                "",7,"",5,9,"","","","",
                "","",9,4,"",6,"","",5,
                2,"",5,"","","",8,6,""]

mid_sudoku_solution = [1, 2, 8, 3, 6, 5, 7, 9, 4,
                       7, 6, 3, 9, 1, 4, 5, 2, 8, 
                       5, 9, 4, 2, 8, 7, 6, 1, 3, 
                       9, 5, 2, 1, 7, 3, 4, 8, 6, 
                       6, 3, 7, 8, 4, 2, 9, 5, 1, 
                       4, 8, 1, 6, 5, 9, 2, 3, 7, 
                       3, 7, 6, 5, 9, 8, 1, 4, 2, 
                       8, 1, 9, 4, 2, 6, 3, 7, 5, 
                       2, 4, 5, 7, 3, 1, 8, 6, 9]

wrong_sudoku = [1,"","","","","","","",1,
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","","",
                "","","","","","","","",""]

conflict_sudoku = [1,"","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "",2,3,"","","","","","",
                    "",4,5,"","","","","","",
                    "",7,6,"","","","","",""]

sudoku_correct_block = ["1","2","3","","","","","","",
                    "6","5","4","","","","","","",
                    "7","8","9","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","",""]

sudoku_block_incomplete = [1,2,3,"","","","","","",
                    6,"",4,"","","","","","",
                    7,8,9,"","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","","",
                    "","","","","","","","",""]

class TestGetIndices(unittest.TestCase):
    def test_get_blocks(self):
        result = solver.get_all_sudoku_blocks()
        expected_result = [(x*3,y*3) for x in range(3) for y in range(3) ]
        self.assertEqual(result,expected_result)
    def test_get_box_indices(self):
        """
        Test if the function returns a list with the correct indices
        """
        indices = (0,0)
        result = solver.get_cell_indices_box(indices)
        expected_result = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        self.assertEqual(result,expected_result)
    def test_get_row_indices(self):
        """
        Test if the funtion returns a list with correct indices of the cells of a row
        """
        index = 3
        result = solver.get_cell_indices_row(index)
        expected_result = [(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8)]
        self.assertEqual(result,expected_result)
    def test_get_col_indices(self):
        """
        Test if the function returns for the start indices of a columns a list of all cell indices 
        of that columns
        """
        index = 3
        result = solver.get_cell_indices_col(index)
        expected_result = [(i,3) for i in range(9)]
        self.assertEqual(result,expected_result)

class TestSudokuPreprocessing(unittest.TestCase):
    def test_list_to_sudoku(self):
        pass

class TestHelpFunction(unittest.TestCase):
    def test_list_intersection(self):
        lst1 = [5]
        lst2 = [1,2,3,4,5,6,7,8,9]
        result = solver.list_intersection(lst1,lst2)
        self.assertEqual(result,[5])


class TestUnitSolve(unittest.TestCase):
    sudoku = solver.Sudoku(sudoku_block_incomplete)
    cells = solver.get_cell_indices_box((0,0))
    block = (0,0)
    def test_get_block(self):
        sudoku = solver.Sudoku(sudoku_correct_block)
        result = sudoku.get_block((0,0))
        expected_result = [1,2,3,6,5,4,7,8,9]
        self.assertEqual(result,expected_result)
    def test_set_cell(self):
        sudoku = solver.Sudoku(sudoku_block_incomplete)
        sudoku.set_cell((1,1),[5])
        result = sudoku.get_block((0,0))
        expected_result = [1,2,3,6,5,4,7,8,9]
        self.assertEqual(self.sudoku.has_conflict,False)
        self.assertEqual(result,expected_result)
    def test_get_incomplete_block(self):
        self.assertEqual(self.sudoku.has_conflict,False)
        result = self.sudoku.get_block((0,0))
        expected_result = [1,2,3,6,[1,2,3,4,5,6,7,8,9],4,7,8,9]
        self.assertEqual(self.sudoku.has_conflict,False)
        self.assertEqual(result,expected_result)
    def test_get_option_unit_missing_one(self):
        self.assertEqual(self.sudoku.has_conflict,False)
        result = self.sudoku.get_options_in_unit(self.cells)
        self.assertEqual(self.sudoku.has_conflict,False)
        expected_result = [5]
        self.assertEqual(result,expected_result)
    def test_place_digits_with_one_option(self):
        sudoku = solver.Sudoku(sudoku_block_incomplete)
        sudoku.set_cell((1,1),[5])
        sudoku.place_digits_with_single_option_in_sudoku_unit(self.cells)
        self.assertEqual(sudoku.has_conflict,False)
    def test_update_unit_options(self):
        options = [5]
        self.assertEqual(self.sudoku.has_conflict,False)
        self.sudoku.update_options_in_unit(self.cells,options)
        result = self.sudoku.get_block(self.block)
        expected_result = [1,2,3,6,5,4,7,8,9]
        self.assertEqual(self.sudoku.has_conflict,False)
        self.assertEqual(result,expected_result)
    def test_unit_solve_block(self):
        sudoku = solver.Sudoku(sudoku_block_incomplete)
        cells = solver.get_cell_indices_box((0,0))
        block = (0,0)
        self.assertEqual(sudoku.has_conflict,False)
        sudoku.unit_solve(cells)
        result = sudoku.get_block(block)
        expected_result = [1,2,3,6,5,4,7,8,9]
        self.assertEqual(sudoku.has_conflict,False)
        self.assertEqual(result,expected_result)
        

class TestSudokuSolver(unittest.TestCase):
    def test_sudoku_unit(self):
        sudoku = solver.Sudoku(sudoku_correct_block)
        block_index = (0,0)
        result= sudoku.correct_block(block_index)
        self.assertEqual(result,True)

    def test_sudoku_solvable_mid(self):
        pass
    def test_sudoku_empty(self):
        sudoku = empty_sudoku
        self.assertEqual(81,len(sudoku))
    def test_sudoku_wrong(self):
        sudoku_as_list = wrong_sudoku
        self.assertEqual(81,len(sudoku_as_list))
    def test_sudoku_unsolvable(self):
        sudoku_as_list = conflict_sudoku 
        self.assertEqual(81,len(sudoku_as_list))


import copy

if __name__=="__main__":
    unittest.main()
    