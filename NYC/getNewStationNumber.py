from dateutil.parser import parse
from utils.dayType import *
from dataAPI.utils import *
from dataAPI.apis import getNearStationList

fileRangeList = [
    [0, 14],
    [50, 67]
]

# timeRange = ['2013-07-01', '2017-09-30']
# trainDataTimeRange = ['2013-07-01', '2016-09-30']
# valDataTimeRange = ['2016-10-01', '2016-12-31']
# testDataRange = ['2017-03-5', '2017-09-30']
#
# date = parse(testDataRange[0])
# dayCounter = 0
# while date <= parse(testDataRange[1]):
#     dateString = date.strftime(dateTimeMode)
#     if isWorkDay(dateString) and isBadDay(dateString) == False:
#         dayCounter += 1
#     date = date + datetime.timedelta(days=1)
# print(dayCounter)

timeRange = ['2013-02-01', '2017-09-30']
stationIDDict = getJsonData('centralStationIDList.json')
centralStationIDList = stationIDDict['centralStationIDList']
allStationIDList = stationIDDict['allStationIDList']

stationRange = fileRangeList[0]
for i in range(stationRange[0], stationRange[1]):
    targetStation = allStationIDList[i*4]
    absStationNumber = [computeDistanceBetweenAB(targetStation, e) for e in getNearStationList(targetStation, 0, 300) if parse(getBuildTime(e)) >= parse(timeRange[0])
                        and parse(getBuildTime(e)) <= parse(timeRange[1])]
    stiStationNumber = [computeDistanceBetweenAB(targetStation, e) for e in getNearStationList(targetStation, 300, 2000) if parse(getBuildTime(e)) >= parse(timeRange[0])
                        and parse(getBuildTime(e)) <= parse(timeRange[1])]
    print(i+1, absStationNumber.__len__(), stiStationNumber.__len__())

stationRange = fileRangeList[1]
for i in range(stationRange[0], stationRange[1]):
    targetStation = allStationIDList[i*4]
    absStationNumber = [computeDistanceBetweenAB(targetStation, e) for e in getNearStationList(targetStation, 0, 300) if parse(getBuildTime(e)) >= parse(timeRange[0])
                        and parse(getBuildTime(e)) <= parse(timeRange[1])]
    stiStationNumber = [computeDistanceBetweenAB(targetStation, e) for e in getNearStationList(targetStation, 300, 2000) if parse(getBuildTime(e)) >= parse(timeRange[0])
                        and parse(getBuildTime(e)) <= parse(timeRange[1])]
    print(i*4, absStationNumber.__len__(), stiStationNumber.__len__())