# A list is symmetric if the first row is the same as the first column,
# the second row is the same as the second column and so on. Write a
# procedure, symmetric, which takes a list as input, and returns the
# boolean True if the list is symmetric and False if it is not.
def symmetric(lista):
    if lista == [] or lista[0] == 1:
        return True
    nlinha = len(lista)
    ncoluna = len(lista[0])
    if nlinha == ncoluna:
        l = 0
        c = 0
        while l < nlinha:
            while c < ncoluna:
                if lista[l][c] != lista[c][l]:
                    return False
                c +=1
            l += 1
        return True
    return False



print symmetric([[1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 3]])
#>>> False

print symmetric([[1,2,3],
                 [2,3,1]])
#>>> False

print symmetric([['cricket', 'football', 'tennis'],
                ['golf']])
