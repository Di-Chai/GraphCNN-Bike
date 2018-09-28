from dataAPI.utils import *
from functools import reduce

if __name__ == '__main__':
    resultDict = getJsonDataFromPath(os.path.join(txtPath, 'stationInDemandData.json'))
    # load the station id on island
    clusterLabelsOnIsland = [19, 5, 2, 26, 17, 12, 22, 29, 11, 6]  # 24, 13,
    fileName = 'HCLUSTER.json'
    clusterResult = getJsonData(fileName)
    stationLabelDict = clusterResult['stationLabelDict']
    stationIDList = getStationIDList()
    # reallocate the cluster name
    stationIDListForClustering = [e for e in stationIDList if stationLabelDict[e] in clusterLabelsOnIsland]

    endTime = parse('2017-09-30')
    targetStationIDList = stationIDListForClustering
    timeSlot = 60

    resultDict = {}

    for i in range(len(targetStationIDList)):
        currentStationID = targetStationIDList[i]

        buildTime = getBuildTime(currentStationID)
        startTime = parse(buildTime)

        # resultDict[currentStationID] = {'buildTime': startTime.strftime(dateTimeMode)}

        stationMinDemandData = getJsonDataFromPath(os.path.join(demandMinDataPath, currentStationID + '.json'))

        totalInList = []
        while startTime <= endTime:
            dateString = startTime.strftime(dateTimeMode)
            print(i, dateString)
            inList = []
            for j in range(24 * 60):
                result = [0, 0]
                resultStation = [None, None]
                if dateString not in stationMinDemandData:
                    pass
                elif str(j) in stationMinDemandData[dateString]['in']:
                    result[0] = stationMinDemandData[dateString]['in'][str(j)]
                    resultStation[0] = stationMinDemandData[dateString]['inStation'][str(j)]
                elif str(j) in stationMinDemandData[dateString]['out']:
                    result[1] = stationMinDemandData[dateString]['out'][str(j)]
                    resultStation[1] = stationMinDemandData[dateString]['outStation'][str(j)]
                inList.append(max(0, result[0]))
            inList = [sum(inList[e: e+timeSlot]) for e in range(len(inList)) if e % timeSlot == 0]
            totalInList.append(inList)

            startTime = startTime + datetime.timedelta(days=1)
        totalInList = reduce(lambda x,y:x+y, totalInList)
        resultDict[currentStationID] = totalInList

    saveJsonDataToPath(resultDict, os.path.join(txtPath, 'stationInDemandData.json'))