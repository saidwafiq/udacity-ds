

def find_element(lista,valor):
    if valor in lista:
        return lista.index(valor)
    return -1



print find_element([1,2,3],3)
print find_element(['alpha','gamma'],'gamma')
