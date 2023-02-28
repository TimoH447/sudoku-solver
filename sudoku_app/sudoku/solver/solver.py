import numpy as np
import copy 

sudoku = [[[""],[2],[""],[8],[""],[1],[7],[""],[""]],
                    [[""],[""],[""],[9],[2],[""],[""],[""],[""]],
                    [[""],[""],[3],[""],[7],[4],[""],[2],[""]],
                    [[""],[6],[""],[""],[9],[""],[""],[""],[7]],
                    [[1],[9],[7],[""],[""],[3],[""],[""],[8]],
                    [[3],[""],[4],[7],[6],[""],[2],[""],[""]],
                    [[5],[""],[""],[6],[""],[""],[""],[7],[""]],
                    [[8],[""],[""],[""],[""],[7],[1],[9],[2]],
                    [[""],[""],[9],[4],[8],[""],[3],[""],[5]]]
def preprocess_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j][0]=="":
                sudoku[i][j]=[k for k in range(1,10)]

def check_sudoku(sudoku):
    pass

def has_conflicts(sudoku):
    #check if a number is twice in a line

    # check if a number is twice in a block
    #check if every number is still possible in the lines and blocks
    return False

def get_cell_indices_row(index):
    return [(index,i) for i in range(9)]

def get_cell_indices_col(index):
    return [(i,index) for i in range(9)]


def get_cell_indices_box(index):
    x,y=index
    cell_indices = []
    for i in range(3):
        for j in range(3):
            cell_indices.append((x+i,y+j))
    return cell_indices

def block_solve(sudoku):
    conflict = False
    for i in range(3):
        x = i*3 
        for j in range(3):
            y=j*3
            checklist = [k for k in range(1,10)]
            options = []
            cells = get_cell_indices_box((x,y))
            for n in range(3):
                for m in range(3):
                    u,v=x+n,y+m
                    if  len(sudoku[u][v])==1 and isinstance(sudoku[u][v][0],int):
                        checklist[int(sudoku[u][v][0]-1)]=0
            for result in checklist:
                if result!=0:
                    options.append(result)
            #comparing possible solutions
            print(x,y)
            print(options)
            for n in range(3):
                for m in range(3):
                    u,v=x+n,y+m
                    remaining_options = []
                    if len(sudoku[u][v])>1:
                        for option in options:
                            if option in sudoku[u][v]:
                                remaining_options.append(option)
                        if remaining_options==[]:
                            conflict = True
                        else:
                            sudoku[u][v]=remaining_options

            # check if a digit is only in one cell, if so the digit is the solution for the cell
            for d in range(1,10):
                digit_count = 0
                for cell in cells:
                    i,j = cell
                    if d in sudoku[i][j]:
                        digit_count+=1
                if digit_count == 0:
                    conflict = True
                elif digit_count ==1:
                    #digit eintragen
                    for cell in cells:
                        i,j = cell
                        if d in sudoku[i][j]:
                            sudoku[i][j]=[d]
    return conflict



def naive_solver(sudoku):
    """
    for area in areas:
        n=check_missing(area)
        if n:
    """
    preprocess_sudoku(sudoku)
    for i in range(5):
        #line_solve_naive(sudoku)
        #column_solve_naive(sudoku)
        block_solve(sudoku)
    return 

def preprocess_list(sudoku_list):
    """This function takes a list with 81 character being either an empty string or '1', ..., '9'.
    It returns list which can be read by the sudoku solver """
    processed_list = []
    for i in range(9):
        row = []
        for j in range(9):
            if sudoku_list[i*9+j] == "":
                row.append([""])
            else:
                row.append([int(sudoku_list[i*9+j])])
        processed_list.append(row)
    preprocess_sudoku(processed_list)
    return processed_list

def revert_sudoku_to_list(sudoku):
    """Reverts a sudoku to a list of 81 elements"""
    reverted_list = []
    for row in sudoku:
        for element in row:
            reverted_list.append(element[0])
    return reverted_list

def same_sudoku(a,b):
    return a==b


def is_unsolved(sudoku) -> bool:
    unsolved = False
    for row in sudoku:
        for cell in row:
            if len(cell)>1:
                return True

def solve_sudoku(sudoku):
    """This function solves a sudoku, if a sudoku is not solvable it returns 'ERROR' """
    preprocess_sudoku(sudoku)
    unsolved = True
    alternatives = []


    while unsolved:
        # if a sudoku has a conlflict than it returns to a state prior guessing a digit
        # if no digits were guessed by the algo and still there is a conflict -> sudoku cannot be solved
        if has_conflicts(sudoku):
            if len(alternatives)==0:
                return "ERROR"
            sudoku = alternatives.pop(-1)

        # copy the current state of the sudoku to see if we get further with solving it without guessing
        temp = copy.deepcopy(sudoku)
        has_conflict = line_solve(sudoku)

        # check if solving without guessing achieved anything
        if same_sudoku(temp,sudoku):
            has_conflict = block_solve(sudoku)
            if same_sudoku(temp,sudoku):
                #alternative = guess_digit(sudoku)
                #alternatives.append(alternative)
                pass
            
        unsolved = is_unsolved(sudoku) 

def get_all_sudoku_blocks():
    """
    returns a list of tuples with the left top corner of all blocks in a sudoku
    """
    return [(x*3,y*3) for x in range(3) for y in range(3) ]

def list_intersection(lst1,lst2):
    return [item for item in lst1 if item in lst2]

class Sudoku:
    def __init__(self,sudoku):
        self.sudoku = preprocess_list(sudoku)
        preprocess_sudoku(self.sudoku)
        self.has_conflict = False
        self.alternatives = []
        self.last_solver = 'guess'
        self.last_sudoku = None

        self.number_of_guesses = 0

    def get_sudoku_as_list(self):
        return revert_sudoku_to_list(self.sudoku)

    def get_block(self,block_index):
        cells = get_cell_indices_box(block_index)
        block = []
        for i,j in cells:
            if len(self.sudoku[i][j])==1:
                block.append(self.sudoku[i][j][0])
            else:
                block.append(self.sudoku[i][j])
        return block
    
    def set_cell(self,cell_index,options):
        i,j = cell_index
        self.sudoku[i][j].clear()
        for option in options:
            self.sudoku[i][j].append(option)

    def is_correct_unit(self,unit_indices):
        unit_numbers = []
        for i,j in unit_indices:
            cell_digit = self.sudoku[i][j]
            if len(cell_digit)==1:
                if cell_digit in unit_numbers:
                    self.has_conflict = True
                    return False
                else:
                    unit_numbers.append(cell_digit)
            else:
                return False
        return True

    def number_has_only_one_option_in_unit(self,cells,number):
        options_count = 0 # the amount of options a number has in the unit(row, col,block)
        for x,y in cells:
            if number in self.sudoku[x][y]:
                options_count += 1
        if options_count == 0:
            self.has_conflict = True
        elif options_count ==1:
            return True
        else:
            return False 


    def correct_row(self,row_index) -> bool:
        """ This function returns true if the row is correctly solved """
        cells = get_cell_indices_row(row_index)
        is_correct = self.is_correct_unit(cells)
        return is_correct
    
    def correct_col(self,col_index):
        cells = get_cell_indices_col(col_index)
        is_correct = self.is_correct_unit(cells)
        return is_correct
    
    def correct_block(self,block_index):
        cells = get_cell_indices_box(block_index)
        is_correct = self.is_correct_unit(cells)
        return is_correct

    def is_solved(self):
        # check rows
        correct_rows = True
        correct_cols = True
        correct_blocks = True
        for i in range(9):
            correct_rows = correct_rows and self.correct_row(i)
        for i in range(9):
            correct_cols = correct_cols and self.correct_col(i)
        
        blocks = get_all_sudoku_blocks()
        for i,j in blocks:
            correct_blocks = correct_blocks and self.correct_block((i,j))
        
        if correct_blocks and correct_cols and correct_rows:
            return True
        else:
            return False

    def guess_digit(self):
        """
        This guesses a digit in the sudoku and then adds another sudoku into the alternatives where the digit 
        is removed from the options in that cell
        """
        for row in self.sudoku:
            for cell in row:
                if len(cell)>1:
                    guess = cell.pop(0)
                    print(guess)
                    alternative = copy.deepcopy(self.sudoku)
                    self.alternatives.append(alternative)
                    cell.clear()
                    cell.append(guess)
                    print('Guessed digit')
                    print(cell)
                    print(self.sudoku)
                    print('End guess')
                    return

    def col_solve(self):
        for i in range(9):
            cells = get_cell_indices_col(i)
            self.unit_solve(cells)

    def row_solve(self):
        # we loop through every row, n is the index of the row
        for i in range(9):
            # we will loop through all cells in the row and remove the cells which are in the line
            cells = get_cell_indices_row(i)
            self.unit_solve(cells)

    def get_options_in_unit(self,cells):
        """
        returns missing digit in the sudoku unit (row,column,block)
        """
        options = [i for i in range(1,10)]
        for i,j in cells:
            if isinstance(self.sudoku[i][j][0],int) and len(self.sudoku[i][j])==1:
                try:
                    options.remove(int(self.sudoku[i][j][0]))
                except:
                    print("There are 2 same digits placed in unit?! Hopfully this sudoku is resetted.")
        return options
    

    def update_options_in_unit(self,cells,options):
        """

        """
        for i,j in cells:
            cell_options = self.sudoku[i][j]

            # dont update cells with only one option
            if len(cell_options)>1:
                rest_options = list_intersection(options,cell_options)
                if rest_options == []:
                    self.has_conflict=True
                else:
                    # update options with the intersection options by the unit and options of the cell
                    self.set_cell((i,j),rest_options)

    def place_digits_with_single_option_in_sudoku_unit(self,cells):
        for d in range(1,10):
            only_once = self.number_has_only_one_option_in_unit(cells,d)
            if only_once:
                #digit eintragen
                for i,j in cells:
                    if d in self.sudoku[i][j]:
                        self.sudoku[i][j].clear()
                        self.sudoku[i][j].append(d)

    def unit_solve(self,cells):
        # first step is to get the left possibilities for digits in this sudoku unit (row,col,block)
        options_in_unit = self.get_options_in_unit(cells)

        # update the options according to the unit
        self.update_options_in_unit(cells,options_in_unit)

        # find and place digits with one option
        self.place_digits_with_single_option_in_sudoku_unit(cells)

    def block_solve(self):
        blocks = get_all_sudoku_blocks()
        for block in blocks:
            cells = get_cell_indices_box(block)            
            self.unit_solve(cells)

    def line_solve(self):
        self.row_solve()
        self.col_solve()

    def next_solving_step(self):
        """
        This function chooses the next solving method: line solve, block solve or guessing a digit
        """
        if self.last_solver == 'line':
            if same_sudoku(self.sudoku,self.last_sudoku):
                print('block')
                self.last_sudoku = copy.deepcopy(self.sudoku)
                self.block_solve()
                self.last_solver = 'block'
            else:
                print('line')
                self.last_sudoku = copy.deepcopy(self.sudoku)
                self.line_solve()
                self.last_solver = 'line'
        elif self.last_solver == 'block':
            if same_sudoku(self.sudoku, self.last_sudoku):
                print('guess')
                self.last_sudoku = copy.deepcopy(self.sudoku)
                self.guess_digit()
                self.number_of_guesses +=1
                self.last_solver = 'guess'
            else:
                print('line')
                self.last_sudoku = copy.deepcopy(self.sudoku)
                self.line_solve()
                self.last_solver = 'line'
        elif self.last_solver == 'guess':
            print('line')
            self.last_sudoku = copy.deepcopy(self.sudoku)
            self.line_solve()
            self.last_solver = 'line'

    def solve_sudoku(self):
        unsolved = True
        max_steps = 1000
        step = 0
        while unsolved:
            print(step)
            print(self.get_sudoku_as_list())
            self.next_solving_step()
            
            if self.is_solved():
                print("sudoku solved")
                unsolved = False

            if self.has_conflict:
                if len(self.alternatives)==0:
                    return "ERROR"
                else:
                    print("fallback since conflict:")
                    print(self.sudoku)
                    self.sudoku = self.alternatives.pop(-1)
                    self.has_conflict = False
            # stop after too many loops
            step+=1
            if step>=max_steps:
                return "ERROR, Runtime"


if __name__=="__main__":
    #trial = 0
    #while trial<4:
    #    line_solve_naive(sudoku)
    #    column_solve_naive(sudoku)
    #    block_solve(sudoku)
    #    trial +=1 
    #print(sudoku)
    unprocessed = ['1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    processed = preprocess_list(unprocessed)
#ADD if sudoku stays the same for a loop then check if solved or not solvable