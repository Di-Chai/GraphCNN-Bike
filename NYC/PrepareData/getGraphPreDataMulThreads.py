from dataAPI.utils import *
from sharedParameters import *



def getGraphPreData(my_rank, stationIDList, timeRange, timeSlot, fileName):
    allTrainData = []
    for stationID in stationIDList:
        print('Threads', my_rank, 'get train data for station', stationID)
        date = parse(timeRange[0])
        endData = parse(timeRange[1])
        stationMinDemandData = getJsonDataFromPath(os.path.join(demandMinDataPath, stationID + '.json'))
        dayIn = []
        dayOut = []
        while date <= endData:
            dateString = date.strftime(dateTimeMode)
            if date < parse(getBuildTime(stationID)):
                date = date + datetime.timedelta(days=1)
                continue
            if isWorkDay(dateString) and isBadDay(dateString) == False:
                inList = []
                outList = []
                for i in range(24 * 60):
                    result = [0, 0]
                    if dateString not in stationMinDemandData:
                        pass
                    elif str(i) in stationMinDemandData[dateString]['in']:
                        result[0] = stationMinDemandData[dateString]['in'][str(i)]
                    elif str(i) in stationMinDemandData[dateString]['out']:
                        result[1] = stationMinDemandData[dateString]['out'][str(i)]
                    inList.append(max(0, result[0]))
                    outList.append(max(0, result[1]))
                dayIn.append([sum(inList[e:e + timeSlot]) for e in range(len(inList)) if
                                      e % timeSlot == 0])
                dayOut.append([sum(outList[e:e + timeSlot]) for e in range(len(outList)) if
                                      e % timeSlot == 0])

            date = date + datetime.timedelta(days=1)
        allTrainData.append([dayIn, dayOut])
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
    centralStationIDList = getJsonData('centralStationIDList.json')['stationIDList']
    n_jobs = 6
    p = Pool()

    avgLength = int(len(centralStationIDList) / n_jobs)
    leftJobs = len(centralStationIDList) % n_jobs
    length = [0] + [avgLength if i >= leftJobs else avgLength + 1 for i in range(n_jobs)]

    for i in range(n_jobs):
        p.apply_async(getGraphPreData,
                      args=(i, centralStationIDList[sum(length[0:i+1]): sum(length[0:i+2])],
                            trainDataTimeRange, timeSlot, 'GraphPreTrainData'), )
    p.close()
    p.join()

    p = Pool()
    for i in range(n_jobs):
        p.apply_async(getGraphPreData,
                      args=(i, centralStationIDList[sum(length[0:i+1]): sum(length[0:i+2])],
                            testDataRange, timeSlot, 'GraphPreTestData'), )
    p.close()
    p.join()

    p = Pool()
    for i in range(n_jobs):
        p.apply_async(getGraphPreData,
                      args=(i, centralStationIDList[sum(length[0:i+1]): sum(length[0:i+2])],
                            valDataTimeRange, timeSlot, 'GraphPreValData'), )
    p.close()
    p.join()

    # get the tem and wind in the father threads
    allTrainData = []
    allTestData = []
    allValData = []
    for i in range(n_jobs):
        allTrainData.append(getJsonData('GraphPreTrainData-%s.json' % i)['allData'])
        os.remove(os.path.join(jsonPath, 'GraphPreTrainData-%s.json' % i))
        allTestData.append(getJsonData('GraphPreTestData-%s.json' % i)['allData'])
        os.remove(os.path.join(jsonPath, 'GraphPreTestData-%s.json' % i))
        allValData.append(getJsonData('GraphPreValData-%s.json' % i)['allData'])
        os.remove(os.path.join(jsonPath, 'GraphPreValData-%s.json' % i))
    allTrainData = reduce(lambda x, y: x + y, allTrainData)
    allTestData = reduce(lambda x, y: x + y, allTestData)
    allValData = reduce(lambda x,y: x+y, allValData)

    trainTemWind = getTemWindList(trainDataTimeRange)
    testTemWind = getTemWindList(testDataRange)
    valTemWind = getTemWindList(valDataTimeRange)

    graphPreData = {
        'allTrainData': allTrainData,
        'allTestData': allTestData,
        'allValData': allValData,
        'trainTem': trainTemWind[0],
        'trainWind': trainTemWind[1],
        'testTem': testTemWind[0],
        'testWind': testTemWind[1],
        'valTem': valTemWind[0],
        'valWind': valTemWind[1]
    }
    saveJsonData(graphPreData, 'GraphPreData.json')