from django.shortcuts import render
from django.views import View
from sudoku.solver import solver
from sudoku.solver import sudoku_api

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

def reset_post(request,context):
    """
    This function handles the post request when pressing the reset button of the webpage
    """
    context['stored_cells'] = get_empty_sudoku_list()
    context['valid'] = True
    return render(request,'sudoku_solver/sudoku-solver.html', context)

def upload_post(request,context):
    """
    This function handles the post request for uploading an image on the website
    """
    # fetch the image from the request
    print(request.FILES)
    image=request.FILES.getlist('photo')
    print(image)
    image = image[0]
    # turn image into the digits of sudoku
    digits = sudoku_api.convert_image_to_digits(image)
    context['stored_cells']= digits
    return render(request, 'sudoku_solver/sudoku-solver.html', context)

def solve_post(request,context):
    """
    This function handles the post request for the solve button
    """
    stored_cells = context["stored_cells"]
    if not valid_input(stored_cells):
        context['valid'] = False
        return render(request, 'sudoku_solver/sudoku-solver.html', context)

    # if input is valid, solve the sudoku
    context['valid'] = True
    sudoku = solver.Sudoku(stored_cells)
    sudoku.solve_sudoku()
    solved =  sudoku.get_sudoku_as_list()
    context['stored_cells'] = solved
    return render(request, 'sudoku_solver/sudoku-solver.html', context)

class SudokuView(View):
    def get(self, request):
        # get request returning a page with an empty sudoku
        stored_cells = get_empty_sudoku_list()
        context = {'stored_cells': stored_cells,
                   'nine': "123456789",
                   'valid': True}
        return render(request, 'sudoku_solver/sudoku-solver.html', context) 

    def post(self, request):
        # saving the inputs in the sudoku
        stored_cells = request.POST.getlist("sudoku-cell")
        context = {'stored_cells': stored_cells,
                   'nine': "123456789"}
        # reset button: resetting sudoku cells with empty list 
        if request.POST.getlist("button")[0] == 'reset':
            return reset_post(request,context)
        # upload image button: input of an sudoku via image
        elif request.POST.getlist("button")[0]=="upload":
            return upload_post(request,context)
        # solve button: 
        else:
            # checking if the input in all sudoku cells is correct
            return solve_post(request,context)
            



