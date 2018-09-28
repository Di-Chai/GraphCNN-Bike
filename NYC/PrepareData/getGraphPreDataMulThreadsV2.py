from dataAPI.utils import *
from sharedParametersV2 import *
from functools import reduce

def getGraphPreData(my_rank, stationIDList, timeRange, timeSlot, fileName, allStationIDList):
    allTrainData = []
    for stationID in stationIDList:
        print('Threads', my_rank, 'get train data for station', stationID)
        date = parse(timeRange[0])
        endData = parse(timeRange[1])
        stationMinDemandData = getJsonDataFromPath(os.path.join(demandMinDataPath, stationID + '.json'))
        dayIn = []
        dayOut = []
        daySum = []
        while date <= endData:
            dateString = date.strftime(dateTimeMode)
            # if date < parse(getBuildTime(stationID)):
            #     date = date + datetime.timedelta(days=1)
            #     continue
            if isWorkDay(dateString) and isBadDay(dateString) == False:
                inList = []
                outList = []
                sumList = []
                for i in range(24 * 60):
                    result = [0, 0]
                    resultStation = [None, None]
                    if dateString not in stationMinDemandData:
                        pass
                    elif str(i) in stationMinDemandData[dateString]['in']:
                        result[0] = stationMinDemandData[dateString]['in'][str(i)]
                        resultStation[0] = stationMinDemandData[dateString]['inStation'][str(i)]
                    elif str(i) in stationMinDemandData[dateString]['out']:
                        result[1] = stationMinDemandData[dateString]['out'][str(i)]
                        resultStation[1] = stationMinDemandData[dateString]['outStation'][str(i)]
                    inList.append(max(0, result[0]))
                    outList.append(max(0, result[1]))
                    sumList.append(max(0, result[0]+result[1]))
                dayIn.append([sum(inList[e:e + timeSlot]) for e in range(len(inList)) if
                                      e % timeSlot == 0])
                dayOut.append([sum(outList[e:e + timeSlot]) for e in range(len(outList)) if
                                      e % timeSlot == 0])
                daySum.append([sum(sumList[e:e + timeSlot]) for e in range(len(sumList)) if
                                      e % timeSlot == 0])
            date = date + datetime.timedelta(days=1)
        if demandType == 'in':
            allTrainData.append(dayIn)
        elif demandType == 'out':
            allTrainData.append(dayOut)
        else:
            allTrainData.append(daySum)
    saveJsonData({'allData': allTrainData}, '%s-%s.json' % (fileName, my_rank))

def getTemWindList(timeRange):
    temList = []
    windList = []
    weatherDict = getJsonData('weatherDict.json')
    temHour = weatherDict['temHour']
    windHour = weatherDict['windHour']
    date = parse(timeRange[0])
    endData = parse(timeRange[1])
    while date <= endData:
        dateString = date.strftime(dateTimeMode)
        if isWorkDay(dateString) and isBadDay(dateString) == False:
            if NO_TEM in temHour[dateString]:
                print(temHour[dateString])
                temHour[dateString] = insertEmptyData(temHour[dateString], NO_TEM)
                print(temHour[dateString])
            temList.append([float(e) for e in temHour[dateString]])
            windList.append([float(e) for e in windHour[dateString]])
        date = date + datetime.timedelta(days=1)
    return temList, windList

if __name__ == '__main__':
    stationIDDict = getJsonData('centralStationIDList.json')
    centralStationIDList = stationIDDict['centralStationIDList']
    allStationIDList = stationIDDict['allStationIDList']

    n_jobs = 12
    avgLength = int(len(allStationIDList) / n_jobs)
    leftJobs = len(allStationIDList) % n_jobs
    length = [0] + [avgLength if i >= leftJobs else avgLength + 1 for i in range(n_jobs)]

    p = Pool()
    for i in range(n_jobs):
        p.apply_async(getGraphPreData,
                      args=(i, allStationIDList[sum(length[0:i + 1]): sum(length[0:i + 2])],
                            timeRange, timeSlotV2, 'GraphPreData', allStationIDList), )
    p.close()
    p.join()

    allData = []
    for i in range(n_jobs):
        allData.append(getJsonData('GraphPreData-%s.json' % i)['allData'])
        os.remove(os.path.join(jsonPath, 'GraphPreData-%s.json' % i))

    allData = reduce(lambda x,y:x+y, allData)

    allData = np.array(allData, dtype=np.float32)

    allDataReshaped = []
    for i in range(allData.shape[1]):
        singleDay = []
        for j in range(allData.shape[2]):
            singleStation = []
            for k in range(allData.shape[0]):
                singleStation.append(int(allData[k, i, j]))
            singleDay.append(singleStation)
        allDataReshaped.append(singleDay)

    temWind = getTemWindList(timeRange)

    saveJsonData({
        'GraphValueMatrix': allDataReshaped,
        'tem': temWind[0],
        'wind': temWind[1]
    }, 'GraphValueMatrix.json')