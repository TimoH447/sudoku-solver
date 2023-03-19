from django.shortcuts import render
from django.views import View
from sudoku.solver import solver

# Create your views here.

def get_empty_sudoku_list():
    return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

def valid_input(grid) -> bool:
    """checks if all list elements are integers, if so it returns true"""
    valid=True
    for element in grid:
        if not(element.isnumeric() and len(element)<=1 or element==''):
            valid = False
    return valid

class SudokuView(View):
    def get(self, request):
        # get request returning a page with an empty sudoku
        stored_cells = get_empty_sudoku_list()
        context = {'stored_cells': stored_cells,
                   'nine': "123456789",
                   'valid': True}
        return render(request, 'sudoku_solver/sudoku-solver.html', context) 

    def post(self, request):
        print(request.POST)
        
        # saving the inputs in the sudoku
        stored_cells = request.POST.getlist("sudoku-cell")
        context = {'stored_cells': stored_cells,
                   'nine': "123456789"}
        # reset button: resetting sudoku cells with empty list 
        if request.POST.getlist("button")[0] == 'reset':
            context['stored_cells'] = get_empty_sudoku_list()
            context['valid'] = True
            return render(request,'sudoku_solver/sudoku-solver.html', context)
        # checking input
        if not valid_input(stored_cells):
            context['valid'] = False
        else:
            # if input is valid, solve the sudoku
            context['valid'] = True
            sudoku = solver.Sudoku(stored_cells)
            sudoku.solve_sudoku()
            solved =  sudoku.get_sudoku_as_list()
            context['stored_cells'] = solved
        return render(request, 'sudoku_solver/sudoku-solver.html', context)



