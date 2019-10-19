def hours2days(h):
    days = h / 24
    hours = h % 24
    return (days,hours)

print hours2days(7)
