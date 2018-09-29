# -*-  coding:utf-8 -*-
from Utils.dayType import *
import random
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import math
from DataAPI.utils import *


dailyDemandDict = safeLoad(getJsonData, 'dailyDemandDict.json')
aggregateDemand = safeLoad(getJsonData, 'aggregateDemand.json')
weatherDict = safeLoad(getJsonData, 'weatherDict.json')
dockDataDict = safeLoad(getJsonData, 'dockDataDict.json')
distanceMatrix = safeLoad(getJsonData, 'distanceMatrix.json')

def getRawBikeDataFileList():
    return sorted([e for e in os.listdir(rawBikeDataPath) if e.endswith('.csv')])

def getNearStationList(stationID, distanceNear, distanceFar):
    resultList = []
    for nearStationID in distanceMatrix[stationID]:
        if distanceMatrix[stationID][nearStationID] < distanceFar and distanceMatrix[stationID][
            nearStationID] > distanceNear:
            resultList.append(nearStationID)
    return resultList


def check_positive(dataList):
    positiveFlag = False
    for e in dataList:
        if e > 0:
            positiveFlag = True
            break
    return positiveFlag

def computeDailyDemand(stationID, date):
    with open(os.path.join(demandDataPath, stationID + '.json'), 'r') as f:
        currentStationDemand = json.load(f)
    try:
        dateDemand = currentStationDemand[date]
    except:
        return EMPTY_DATA
    hourDemandTotal = 0
    for hour in dateDemand:
        for type in dateDemand[hour]:
            for targetStation in dateDemand[hour][type]:
                hourDemandTotal += dateDemand[hour][type][targetStation]
    return hourDemandTotal


def getDailyDemand(stationID, date):
    try:
        return dailyDemandDict[stationID][date]
    except:
        return EMPTY_DATA


def getStationBuildTime_String(stationID):
    return stationAppearTimeDict[stationID][0]


def getStationBuildTime_DataTime(stationID):
    return parse(stationAppearTimeDict[stationID][0])


def getDailyAggregateDemand(dateString):
    dateStringNormal = parse(dateString).strftime(dateTimeMode)
    try:
        if dateStringNormal in aggregateDemand['workDay']:
            return aggregateDemand['workDay'][dateStringNormal]
        else:
            return aggregateDemand['holiday'][dateStringNormal]
    except:
        # print('Can not find %s in aggregateDemand' % dateStringNormal)
        return -1


def getAverageDailyAggregateDemand(dateRange):
    dateRangeNormal = [parse(e) for e in dateRange]
    demandSum = []
    for date in dateRangeNormal:
        demandSum.append(getDailyAggregateDemand(date.strftime(dateTimeMode)))
    if check_positive(demandSum):
        averageDemand = get_positive_mean(demandSum)
    else:
        averageDemand = np.mean(demandSum)
    return averageDemand


def getDateList(dayType):
    return list(aggregateDemand[dayType].keys())


def getTemperature(dateString):
    temperature = weatherDict['temperature']
    try:
        return temperature[dateString]
    except:
        return NO_TEM


def getWindSpeed(dateString):
    windSpeed = weatherDict['wind']
    try:
        return windSpeed[dateString]
    except:
        return EMPTY_DATA

def deleteFileFromPath(path, startWith='', endWith=''):
    fileToDelete = [e for e in os.listdir(path) if e.startswith(startWith) and e.endswith(endWith)]
    for e in fileToDelete:
        os.remove(os.path.join(path, e))


def get_demand_normal_mean(valueList):
    if check_positive(valueList):
        valueListPositive = [e for e in valueList if e > 0]
        valueListMean = np.mean(valueListPositive)
    else:
        valueListMean = 0
        if -1 in valueList:
            valueListMean += -1
        if -2 in valueList:
            valueListMean += -2
    return float('%.2f' % valueListMean)


def get_normal_mean(valueList):
    if check_positive(valueList):
        valueListPositive = [e for e in valueList if e > 0]
        valueListMean = np.mean(valueListPositive)
    else:
        valueListMean = valueListMean = np.mean(valueList)
    return valueListMean


def get_tem_normal_mean(valueList):
    if check_tem_normal(valueList):
        valueListPositive = [e for e in valueList if e != NO_TEM]
        valueListMean = np.mean(valueListPositive)
    else:
        valueListMean = NO_TEM
    return float('%.2f' % float(valueListMean))


def partitionByTime(timeRange, valueDateList, valueList):
    if valueList.__len__() != valueDateList.__len__():
        print("aggregateByTime length error")
        return -1
    # add together
    partitionValueList = [[] for _ in range(timeRange.__len__())]
    for dateCounter in range(valueDateList.__len__()):
        date = valueDateList[dateCounter]
        for i in range(timeRange.__len__()):
            if date in timeRange[i]:
                partitionValueList[i].append(valueList[dateCounter])
    return partitionValueList


def cutListByTime(valueList, valueListDate, cutTime):
    valueListDate = [parse(e) for e in valueListDate]
    cutTime = parse(cutTime)
    valueList0 = []
    valueList1 = []
    for i in range(valueListDate.__len__()):
        if valueListDate[i] < cutTime:
            valueList0.append(valueList[i])
        elif valueListDate[i] > cutTime:
            valueList1.append(valueList[i])
    return valueList0, valueList1


def getTotalDocks(stationID, dateString):
    dockDict = dockDataDict['dockDict']
    if stationID not in dockDict:
        print('Cannot find %s in dockData' % stationID)
        return EMPTY_DATA
    else:
        try:
            totalDocks = float(dockDict[stationID][dateString][2])
        except:
            # print('Cannot find %s for %s in dockData' % (dateString, stationID))
            return EMPTY_DATA
        return totalDocks


def get_distanceMatrix():
    # load data
    stationInfo = getJsonData('stationAppearTime.json')
    stationIDList = getStationIDList()
    # compute the distance matrix
    distanceMatrix = []
    for row in stationIDList:
        rowStation = stationInfo[row]
        distance = []
        for col in stationIDList:
            colStation = stationInfo[col]
            distance.append(haversine(float(rowStation[2]), float(rowStation[1]),
                                      float(colStation[2]), float(colStation[1])))
        distanceMatrix.append(distance)
    return distanceMatrix


def computeAvailableStationNumbers(stationList, dateStringList):
    availableStationNumbers = []
    for dateString in dateStringList:
        count = 0
        for stationID in stationList:
            demand = getDailyDemand(stationID, dateString)
            if demand != EMPTY_DATA:
                count += 1
        availableStationNumbers.append(count)
    return np.mean(availableStationNumbers)


def summarizeDemand(valueList):
    weeklyDemand = []
    for e in valueList:
        weeklyDemand.append(get_demand_normal_mean(e))
    return weeklyDemand


def summarizeTem(valueList):
    weeklyTem = []
    for e in valueList:
        weeklyTem.append(get_tem_normal_mean(e))
    return weeklyTem


def divideList(valueList, gap):
    resultList = []
    valueListLength = valueList.__len__()
    divideTimes = int(valueListLength / gap)
    for i in range(divideTimes):
        resultList.append(valueList[i * gap: (i + 1) * gap])
    return resultList


def computeAggregateDailyDemandOfPartStation(stationIDList, dateString):
    demandResult = []
    for stationID in stationIDList:
        demandResult.append(getDailyDemand(stationID, dateString))
    return get_demand_normal_mean(demandResult)


# def computeAggregateDemandOfPartStationForTimeRange(stationIDList, dateStringList):
#     demandList = []
#     for dateString in dateStringList:
#         demandList.append(computeAggregateDemandOfPartStationForTimeRange(stationIDList, dateString))
#     return get_demand_normal_mean(demandList)

def getWeeklyDemandForTimeRange(stationID, dateStringList):
    workday = []
    holiday = []
    all = []
    for dateString in dateStringList:
        if stationID == None:
            currentDemand = getDailyAggregateDemand(dateString)
        elif type(stationID) == list:
            currentDemand = computeAggregateDailyDemandOfPartStation(stationID, dateString)
        else:
            currentDemand = getDailyDemand(stationID, dateString)
        # 去除坏天气
        if isBadDay(dateString):
            if currentDemand != EMPTY_DATA:
                currentDemand = BAD_WEATHER
            else:
                currentDemand += BAD_WEATHER
        if isWorkDay(dateString):
            workday.append(currentDemand)
        else:
            holiday.append(currentDemand)
        all.append(currentDemand)

    workday_cut = divideList(workday, 5)
    holiday_cut = divideList(holiday, 2)
    all_cut = divideList(all, 7)

    workday_weekly = summarizeDemand(workday_cut)
    holiday_weekly = summarizeDemand(holiday_cut)
    all_weekly = summarizeDemand(all_cut)

    return all_weekly, workday_weekly, holiday_weekly


def getWeeklyTemForTimeRange(dateStringList):
    tem = []
    for dateString in dateStringList:
        tem.append(getTemperature(dateString))
    tem_cut = divideList(tem, 7)
    tem_weekly = summarizeTem(tem_cut)
    return tem_weekly


def isGoodData_WeeklyDemand(valueList):
    dataFlag = False
    if check_positive(valueList):
        for e in valueList:
            if e > 5:
                dataFlag = True
                break
        return dataFlag
    else:
        return dataFlag

def getStationIDListBeforeDate(dateString):
    allStationIDList = getStationIDList()
    resultList = []
    dateDateTime = parse(dateString)
    for stationID in allStationIDList:
        buildTime = parse(getBuildTime(stationID))
        if buildTime < dateDateTime:
            resultList.append(stationID)
    return resultList


def BoxplotOutlier(valueList, times):
    valueList.sort()
    valueListLength = valueList.__len__()
    quarterLength = int(valueListLength / 4)
    upperQuarter = float(valueList[quarterLength])
    lowerQuarter = float(valueList[valueListLength - 1 - quarterLength])
    outlierList = []
    for i in range(valueListLength):
        pointer = valueListLength - 1 - i
        if valueList[pointer] > upperQuarter * times or valueList[pointer] < lowerQuarter / times:
            outlierList.append(valueList[pointer])
            del valueList[pointer]
    return valueList, outlierList


def getPositivePart(dataList):
    positivePart = []
    for e in dataList:
        if e > 0:
            positivePart.append(e)
    return positivePart


def compute_conf_interval(valueList, confidence):
    valueList.sort(reverse=True)
    alpth = 1 - confidence
    alpth_half = alpth / 2.0
    upperLimit = int(valueList.__len__() * alpth_half)
    return (valueList[upperLimit], valueList[valueList.__len__() - 1 - upperLimit])


def BootstrapVariance(sampleTime, confidence, valueList):
    conf_interval_lower = []
    conf_interval_upper = []
    X = [e + 1 for e in range(valueList.__len__())]
    Y = []
    for i in range(valueList.__len__()):
        currentValueList = valueList[i]
        currentValueListPositive, outlierList = BoxplotOutlier(getPositivePart(currentValueList), 2.0)
        Y.append(get_normal_mean(currentValueListPositive))
        print('%s outlier number:%s' % (i + 1, outlierList.__len__()))
        sampleLength = currentValueListPositive.__len__()
        sampleMeanList = []
        for i in range(sampleTime):
            oneSample = []
            for j in range(sampleLength):
                oneSample.append(random.choice(currentValueListPositive))
            sampleMeanList.append(get_normal_mean(oneSample))
        upper, lower = compute_conf_interval(sampleMeanList, confidence)
        conf_interval_lower.append(get_normal_mean(sampleMeanList) - lower)
        conf_interval_upper.append(upper - get_normal_mean(sampleMeanList))
    currentLength = Y.__len__()
    for i in range(Y.__len__()):
        pointer = currentLength - 1 - i
        if Y[pointer] < 0:
            del Y[pointer]
            del X[pointer]
    return X, Y, conf_interval_upper, conf_interval_lower


# def errorBarInfo(valueList, confidence):
#     conf_interval_lower = []
#     conf_interval_upper = []
#     X = [e + 1 for e in range(valueList.__len__())]
#     Y = []
#     for i in range(valueList.__len__()):
#         currentValueList = valueList[i]
#         upper, lower = compute_conf_interval(valueList, confidence)
#         conf_interval_lower.append(get_normal_mean(valueList) - lower)
#         conf_interval_upper.append(upper - get_normal_mean(valueList))

def normalizeList(valueList):
    valueListNPArray = np.array(valueList, dtype=np.float32)
    # 标准化test data
    valueListNPArray -= np.mean(valueListNPArray, axis=0)
    valueListNPArray /= np.std(valueListNPArray, axis=0)
    return list(valueListNPArray)


def adfTest(timeSeries, maxlags=None, printFlag=False):
    t = sm.tsa.stattools.adfuller(timeSeries, maxlag=maxlags)
    output = pd.DataFrame(
        index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
               "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    if printFlag:
        print(output)
    return t


def plot_acf_and_pacf(timeSeries, lags=20):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeSeries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeSeries, lags=lags, ax=ax2)
    plt.show()


def reverseList(valueList):
    resultList = []
    for i in range(valueList.__len__()):
        resultList.append(valueList[valueList.__len__() - i - 1])
    return resultList


def getWorkdayTimeRangeList(targetTime, leftTimeLimit, direction, length, gap=7):
    targetStationBuildTime = parse(leftTimeLimit)
    targetTimeD = parse(targetTime)
    resultList = []
    direction = -1 if direction < 0 else 1
    stopFlag = False
    for i in range(length):
        oneTimeRange = []
        for e in range(1 + i * gap, 1 + (i + 1) * gap):
            date = targetTimeD + datetime.timedelta(days=e * direction)
            if date < targetStationBuildTime:
                stopFlag = True
                break
            if isWorkDay(date.strftime(dateTimeMode)):
                oneTimeRange.append(date)
        if stopFlag:
            break
        else:
            resultList.append([e.strftime(dateTimeMode) for e in oneTimeRange])
    if direction < 0:
        resultList = reverseList(resultList)
    return resultList


def getTimeRangeList(targetTime, leftTimeLimit, direction, length, gap=7):
    targetStationBuildTime = parse(leftTimeLimit)
    targetTimeD = parse(targetTime)
    resultList = []
    direction = -1 if direction < 0 else 1
    stopFlag = False
    for i in range(length):
        oneTimeRange = []
        for e in range(1 + i * gap, 1 + (i + 1) * gap):
            date = targetTimeD + datetime.timedelta(days=e * direction)
            if date < targetStationBuildTime:
                stopFlag = True
                break
            oneTimeRange.append(date)
        if stopFlag:
            break
        else:
            resultList.append([e.strftime(dateTimeMode) for e in oneTimeRange])
    if direction < 0:
        resultList = reverseList(resultList)
    return resultList


def d_diff(value, valueList):
    resultList = []
    for i in range(valueList.__len__()):
        resultList.append(value + sum(valueList[:i + 1]))
    return resultList


def getCurrentDocksNumber(stationID):
    if os.path.isfile(os.path.join(jsonPath, 'station_status.json')) == False:
        post_url = 'https://api-core.citibikenyc.com/gbfs/en/station_status.json'
        response = requests.get(url=post_url)
        try:
            result_json = json.loads(response.text)
        except Exception as e:
            print(response.status_code)
            print(e)
            return -1
        dataList = result_json['data']['stations']
        formattedDataDict = {}
        for record in dataList:
            formattedDataDict[record['station_id']] = record
        result_json['data']['stations'] = formattedDataDict
        saveJsonData(result_json, 'station_status.json')

    station_status = getJsonData('station_status.json')
    '''
    Example:
    {'eightd_has_available_keys': False,
     'is_installed': 1,
     'is_renting': 1,
     'is_returning': 1,
     'last_reported': 1515990136,
     'num_bikes_available': 3,
     'num_bikes_disabled': 0,
     'num_docks_available': 20,
     'num_docks_disabled': 0,
     'station_id': '3070'}'''
    try:
        stationInfo = station_status['data']['stations'][stationID]
    except:
        print('<getCurrentDocksNumber>(Key error : %s)' % stationID)
        return -1
    return stationInfo['num_bikes_available'] + stationInfo['num_docks_available']


def getDockNumberFroTimeRange(stationID, dateStringList):
    dockNumberList = []
    for dateString in dateStringList:
        dockNumber = getTotalDocks(stationID, dateString)
        if dockNumber < 0:
            dockNumber = getCurrentDocksNumber(stationID)
        dockNumberList.append(dockNumber)
    return get_normal_mean(dockNumberList)

def getMeanDemandForTimeRangeList(stationID, timeRangeList):
    return [get_demand_normal_mean([getDailyDemand(stationID, date) for date in timeRange])
            for timeRange in timeRangeList]


def getMeanDemandForTimeRange(stationID, timeRange):
    a = [getDailyDemand(stationID, date) for date in timeRange]
    return get_demand_normal_mean([getDailyDemand(stationID, date) for date in timeRange])


def getAggMeanDemandForTimeRange(timeRangeList):
    return [get_demand_normal_mean([getDailyAggregateDemand(date) for date in timeRange])
            for timeRange in timeRangeList]


def countNewStationNumbersForTimeRange(dateStringList, stationIDList):
    count = 0
    dateList = [parse(e) for e in dateStringList]
    minDate = min(dateList)
    maxDate = max(dateList)
    newStationList = []
    for stationID in stationIDList:
        buildTime = parse(getBuildTime(stationID))
        if buildTime > minDate and buildTime < maxDate:
            count += 1
            newStationList.append(stationID)
    return count


def getStationBuildBefore(dateStringList, stationIDList):
    resultList = []
    minDate = min([parse(e) for e in dateStringList])
    for stationID in stationIDList:
        if parse(getBuildTime(stationID)) <= minDate:
            resultList.append(stationID)
    return resultList


server = False
if server:
    MinDemandDict = {stationID: getJsonDataFromPath(os.path.join(demandMinDataPath, stationID + '.json'))
                     for stationID in getStationIDList()}

def getMinDemand(stationID, dateString, minString):
    result = [EMPTY_DATA, EMPTY_DATA]
    try:
        stationMinDemandData = MinDemandDict[stationID]
    except:
        stationMinDemandData = getJsonDataFromPath(os.path.join(demandMinDataPath, stationID + '.json'))
    if dateString not in stationMinDemandData:
        return result
    elif minString in stationMinDemandData[dateString]['in']:
        result[0] = stationMinDemandData[dateString]['in'][minString]
    elif minString in stationMinDemandData[dateString]['out']:
        result[1] = stationMinDemandData[dateString]['out'][minString]
    return result

def findMarginPoint(pointList, fineness=1, ratio=0.5):
    centerPoint = [np.mean([e[0] for e in pointList]), np.mean([e[1] for e in pointList])]
    # create the polar coordinates
    polarCoordinates = []
    for i in range(pointList.__len__()):
        point = pointList[i]
        distance = ((point[0]-centerPoint[0])**2+(point[1]-centerPoint[1])**2)**0.5
        cosValue = (point[0]-centerPoint[0]) / distance
        degree = math.acos(cosValue)
        if point[1] < centerPoint[1]:
            degree = 2 * math.pi - degree
        polarCoordinates.append([distance, degree])
    marginPointList = []
    startDegree = 0
    maxDistance = -1
    while startDegree < 360:
        degreeInterval = [math.radians(startDegree), math.radians(startDegree + fineness)]
        pointInInterval = []
        for i in range(polarCoordinates.__len__()):
            if polarCoordinates[i][1] >= degreeInterval[0] and polarCoordinates[i][1] < degreeInterval[1]:
                pointInInterval.append([i, polarCoordinates[i]])
        pointInIntervalSorted = sorted(pointInInterval, key=lambda x: x[1][0], reverse=True)
        if len(pointInIntervalSorted) > 0:
            if maxDistance == -1:
                maxDistance = pointInIntervalSorted[0][1][0]
            if pointInIntervalSorted[0][1][0] / maxDistance > ratio:
                marginPoint = pointList[pointInIntervalSorted[0][0]]
                marginPointList.append(marginPoint)
                maxDistance = pointInIntervalSorted[0][1][0]
        startDegree = startDegree + fineness

    return marginPointList

def findMarginPointV2(pointList, distanceToLine=0.001):
    marginPointIndex = []
    normOp = lambda x: (x[0]**2 + x[1]**2)**0.5
    distanceOp = lambda x,y: ((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5
    AnticlockwiseRotate90Op = lambda x: [-x[1], x[0]]
    vectorOp = lambda x,y: [y[0]-x[0], y[1]-x[1]]
    vectorMulOp = lambda x,y: x[0]*y[0] + x[1]*y[1]
    vectorAngleOp = lambda x,y: vectorMulOp(x,y)/(normOp(x)*normOp(y))
    distanceToVectorOp = lambda x, v0, vector: abs(1-vectorAngleOp(vectorOp(v0, x), vector)**2)**0.5 * normOp(vectorOp(v0, x))
    centerPoint = [np.mean([e[0] for e in pointList]), np.mean([e[1] for e in pointList])]
    # 1 find the max distance point
    distanceToCenter = [distanceOp(e, centerPoint) for e in pointList]
    maxIndex = distanceToCenter.index(max(distanceToCenter))
    marginPointIndex.append(maxIndex)
    maxPoint = pointList[maxIndex]
    # get the vector (AnticlockwiseRotate90)
    vector0 = AnticlockwiseRotate90Op(vectorOp(maxPoint, centerPoint))
    # get the next 2 points
    degreeList = [] # cosValue
    for i in range(pointList.__len__()):
        if i == maxIndex:
            continue
        degreeList.append([i, vectorAngleOp(vector0, vectorOp(maxPoint, pointList[i]))])
    sortedDegreeList = sorted(degreeList, key=lambda x: x[1], reverse=True)
    # 2 get the start point and the end point
    startPointIndex = sortedDegreeList[0][0]
    endpointIndex = sortedDegreeList[-1][0]
    marginPointIndex.append(startPointIndex)
    marginPointIndex.append(endpointIndex)

    nearPointIndexToLine = []
    for i in range(pointList.__len__()):
        if i == maxIndex or i == startPointIndex or i == endpointIndex:
            continue
        if distanceToVectorOp(pointList[i], maxPoint, vectorOp(maxPoint, pointList[startPointIndex])) < distanceToLine:
            nearPointIndexToLine.append(i)
        if distanceToVectorOp(pointList[i], maxPoint, vectorOp(maxPoint, pointList[endpointIndex])) < distanceToLine:
            nearPointIndexToLine.append(i)
    for index in nearPointIndexToLine:
        if index not in marginPointIndex:
            marginPointIndex.append(index)

    currentPointIndex = -1
    lastPointIndex = -1
    while currentPointIndex != endpointIndex:
        print(marginPointIndex.__len__())
        if currentPointIndex == -1:
            currentPointIndex = startPointIndex
            lastPointIndex = maxIndex
        currentPoint = pointList[currentPointIndex]
        lastPoint = pointList[lastPointIndex]
        vector0 = vectorOp(lastPoint, currentPoint)
        degreeList = []  # cosValue
        for i in range(pointList.__len__()):
            if i == currentPointIndex:
                continue
            degreeList.append([i, vectorAngleOp(vector0, vectorOp(currentPoint, pointList[i]))])
        sortedDegreeList = sorted(degreeList, key=lambda x: x[1], reverse=True)
        nextPointIndex = sortedDegreeList[0][0]

        nearPointIndexToLine = []
        distanceToLineList = []
        for i in range(pointList.__len__()):
            if i == currentPointIndex or i == nextPointIndex:
                continue
            if distanceToVectorOp(pointList[i], currentPoint,vectorOp(currentPoint, pointList[nextPointIndex])) < distanceToLine:
                nearPointIndexToLine.append(i)
        lastPointIndex = currentPointIndex
        if nextPointIndex not in marginPointIndex:
            marginPointIndex.append(nextPointIndex)
        for index in nearPointIndexToLine:
            if index not in marginPointIndex:
                marginPointIndex.append(index)
        currentPointIndex = nextPointIndex

    return marginPointIndex

