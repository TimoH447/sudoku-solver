from django.shortcuts import render
from django.views import View

# Create your views here.

def valid_input(grid) -> bool:
    """checks if all list elements are integers, if so it returns true"""
    valid=True
    for element in grid:
        if not(element.isnumeric() and len(element)<=1 or element==''):
            valid = False
    return valid

class SudokuView(View):
    def get(self, request):
        stored_cells = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
        context = {'stored_cells': stored_cells,
                   'nine': "123456789",
                   'valid': True}
        return render(request, 'sudoku_solver/sudoku-solver.html', context) 

    def post(self, request):
        print(request.POST)
        stored_cells = request.POST.getlist("sudoku-cell")
        context = {'stored_cells': stored_cells,
                   'nine': "123456789"}
        if not valid_input(stored_cells):
            context['valid'] = False
        else:
            context['valid'] = True
        return render(request, 'sudoku_solver/sudoku-solver.html', context)



