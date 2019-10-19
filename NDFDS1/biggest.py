def greatest(list_of_numbers):
    big = 0
    for e in list_of_numbers:
        if e > big:
            big = e
    return big

print greatest([4,23,1])
