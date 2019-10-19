###
### Define a simple nextDay procedure, that assumes
### every month has 30 days.
###
### For example:
###    nextDay(1999, 12, 30) => (2000, 1, 1)
###    nextDay(2013, 1, 30) => (2013, 2, 1)
###    nextDay(2012, 12, 30) => (2013, 1, 1)  (even though December really has 31 days)
###
def isLeapYear(year):
    if year % 4 == 0:
        if year % 100 == 0 and year % 400 != 0:
            return False
        return True
    return False

def daysInMonth (year,month):
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    if month == 2:
        if isLeapYear(year) is True:
            return 29
        return 28
    return 31

def nextDay(year, month, day):
    if day < daysInMonth(year,month):
        return year, month, day + 1
    else:
        if month < 12:
            return year, month + 1, 1
        else:
            return year + 1, 1, 1

###def nextDay(year, month, day):
###    if day < 30:
###        return year, month, day + 1
###    else:
###        if month < 12:
###            return year, month + 1, 1
###        else:
###            return year + 1, 1, 1

def dateIsBefore(year1,month1,day1,year2,month2,day2):
    if year2 > year1:
        return True
    if year2 == year1:
        if month2 > month1:
            return True
        if month2 == month1:
            return day2 > day1
    return False

def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    assert not dateIsBefore(year2, month2, day2,year1, month1, day1)
    days = 0
    while dateIsBefore(year1, month1, day1, year2, month2, day2):
         year1,month1,day1 = nextDay(year1,month1,day1)
         days +=1
    return days

def test():
    test_cases = [((2012,9,30,2012,10,30),30),
                  ((2012,1,1,2013,1,1),366),
                  ((2012,9,1,2012,9,4),3)]

    for (args, answer) in test_cases:
        result = daysBetweenDates(*args)
        if result != answer:
            print "Test with data:", args, "failed"
        else:
            print "Test case passed!"

test()
