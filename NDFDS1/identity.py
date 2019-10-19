def is_identity_matrix(matrix):
    if len(matrix) == len(matrix[0]):
        l,c,czero = 0,0,0
        # verificar 0
        for e in matrix:
            for i in e:
                if i == 0:
                    czero +=1
        if czero + len(matrix) != len(matrix) * len(matrix):
            return False
        # verificar diagonal1
        while l < len(matrix):
            if matrix[l][c] == 1:
                c +=1
                l += 1
            else:
                return False
        return True
    return False

# Test Cases:

matrix1 = [[1,0,0,0],
           [0,1,0,0],
           [0,0,1,0],
           [0,0,0,1]]
print is_identity_matrix(matrix1)
#>>>True

matrix2 = [[1,0,0],
           [0,1,0],
           [0,0,0]]

print is_identity_matrix(matrix2)
#>>>False

matrix3 = [[2,0,0],
           [0,2,0],
           [0,0,2]]

print is_identity_matrix(matrix3)
#>>>False

matrix4 = [[1,0,0,0],
           [0,1,1,0],
           [0,0,0,1]]

print is_identity_matrix(matrix4)
#>>>False

matrix5 = [[1,0,0,0,0,0,0,0,0]]

print is_identity_matrix(matrix5)
#>>>False

matrix6 = [[1,0,0,0],
           [0,1,0,1],
           [0,0,1,0],
           [0,0,0,1]]

print is_identity_matrix(matrix6)
#>>>False

matrix7 = [[1, -1, 1],
           [0, 1, 0],
           [0, 0, 1]]
print is_identity_matrix(matrix7)
#>>>False
