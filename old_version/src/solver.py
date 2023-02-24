import numpy as np
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

def solve_sudoku(sudoku):

    solved_sudoku = np.zeros((9,9))
    return solved_sudoku

def check_sudoku(sudoku):
    pass

def line_solve_naive(sudoku):
    for n in range(9):
        checklist = [i for i in range(1,10)]
        options = []
        for i in range(9):
            if isinstance(sudoku[n][i][0],int) and len(sudoku[n][i])==1:
                checklist[int(sudoku[n][i][0]-1)]=0
        for result in checklist:
            if result!=0:
                options.append(result)
        if len(options)==1:
            print(f"Found a correct number {options[0]}")
        for i in range(9):
            remaining_options = []
            if len(sudoku[n][i])>1:
                for option in options:
                    if option in sudoku[n][i]:
                        remaining_options.append(option)
                sudoku[n][i]=remaining_options
            

def column_solve_naive(sudoku):
    for n in range(9):
        checklist = [i for i in range(1,10)]
        options = []
        for i in range(9):
            if isinstance(sudoku[i][n][0],int) and len(sudoku[i][n])==1:
                checklist[int(sudoku[i][n][0]-1)]=0
        for result in checklist:
            if result!=0:
                options.append(result)
        if len(options)==1:
            print(f"Found a correct number {options[0]}")
        #comparing possible solutions
        for i in range(9):
            remaining_options = []
            if len(sudoku[i][n])>1:
                for option in options:
                    if option in sudoku[i][n]:
                        remaining_options.append(option)
                sudoku[i][n]=remaining_options

def block_solve(sudoku):
    for i in range(3):
        x = i*3 
        for j in range(3):
            y=j*3
            checklist = [k for k in range(1,10)]
            options = []
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
                        sudoku[u][v]=remaining_options

def naive_solver(sudoku):
    """
    for area in areas:
        n=check_missing(area)
        if n:
    """
    line_solve_naive(sudoku)
    return 

if __name__=="__main__":
    trial = 0
    preprocess_sudoku(sudoku)
    while trial<4:
        line_solve_naive(sudoku)
        column_solve_naive(sudoku)
        block_solve(sudoku)
        trial +=1 
    print(sudoku)

#ADD if sudoku stays the same for a loop then check if solved or not solvable