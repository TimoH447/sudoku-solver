from django import template

register = template.Library()

@register.filter
def chunk(list_, n):
    """
    Splits a list into sub-lists of a given length.
    """
    return [list_[i:i+n] for i in range(0, len(list_), n)]

@register.filter
def enumerate_list(value):
    return enumerate(value)

@register.filter(name='argument_helper')
def argument_helper(list_,i):
    return list_,i
@register.filter
def sudoku_value(tuple,j):
    list_, i = tuple
    i=int(i)-1
    j=int(j)-1
    index = i * 9 + j
    value = str(list_[index])
    return value