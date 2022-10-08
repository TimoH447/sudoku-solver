import pytest
import unittest

from src.solver import solve_sudoku

def test_solve_sudoku():
    assert solve_sudoku(10).shape == (9,9)
    
