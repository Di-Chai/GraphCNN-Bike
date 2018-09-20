import datetime
from matplotlib import pyplot as plt
dateTimeMode = '%Y-%m-%d'



def getWeekTimeRange(targetDate, k):
    # 找到前后k周的时间范围
    weekTimeRange = []
    weekTimeRangeBefore = []
    weekTimeRangeAfter = []
    currentDate = targetDate
    currentDate = currentDate + datetime.timedelta(days=-1)
    for i in range(k):
        weekTime = []
        while currentDate.weekday() != 4:
            currentDate = currentDate + datetime.timedelta(days=-1)
        for i in range(5):
            weekTime.append(currentDate.strftime(dateTimeMode))
            currentDate = currentDate + datetime.timedelta(days=-1)
        weekTimeRangeBefore.insert(0, weekTime)
        weekTimeRange.insert(0, weekTime)
    currentDate = targetDate
    currentDate = currentDate + datetime.timedelta(days=1)
    for i in range(k):
        weekTime = []
        while currentDate.weekday() != 0:
            currentDate = currentDate + datetime.timedelta(days=1)
        for i in range(5):
            weekTime.append(currentDate.strftime(dateTimeMode))
            currentDate = currentDate + datetime.timedelta(days=1)
        weekTimeRangeAfter.append(weekTime)
        weekTimeRange.append(weekTime)
    return weekTimeRange

def cutList(valueList):
    if valueList.__len__() % 2 != 0:
        print('cutList length error')
        return -1
    else:
        valueListLength = int(valueList.__len__()/2)
        return valueList[0: valueListLength], valueList[valueListLength : ]

# 给柱状图添加text
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2-0.3, height, '%s' % float(height))

def positiveDivide(a, b):
    if b == 0:
        b = 0.001
    if a < 0 and b < 0:
        return float('%.3f'%(-a / b))
    else:
        return float('%.3f'%(a / b))

def getDistinctElements(valueList):
    resultList = []
    for e in valueList:
        if e not in resultList:
            resultList.append(e)
    return resultList

def get_conf_part(valueList, confidence):
    valueList.sort(reverse=True)
    alpth = 1 - confidence
    alpth_half = alpth / 2.0
    upperLimit = int(valueList.__len__() * alpth_half)
    return valueList[upperLimit:valueList.__len__() - 1 - upperLimit]

def subtractList(valueList1, valueList2):
    resultList = []
    for e in valueList1:
        if e not in valueList2:
            resultList.append(e)
    return resultList

def cutTimeSeries(seriesList, T):
    pointerStart = 0
    resultList = []
    for i in range(seriesList.__len__()-1):
        if (seriesList[i+1] - seriesList[i]).days > T:
            resultList.append(seriesList[pointerStart: i+1])
            pointerStart = i + 1
    return resultList
