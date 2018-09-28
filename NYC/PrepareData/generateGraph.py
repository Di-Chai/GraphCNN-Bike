from dataAPI.utils import *
from functools import reduce
from scipy.stats import pearsonr

def checkZero(valueList):
    result = True
    for e in valueList:
        if e != 0:
            return False
    return result

if __name__ == '__main__':
    stationIDDict = getJsonData('centralStationIDList.json')
    centralStationIDList = stationIDDict['centralStationIDList']
    allStationIDList = stationIDDict['allStationIDList']

    # distance graph
    # distanceThreshold = 500
    graphMatrix = [[0 for _ in range(len(allStationIDList))] for _ in range(len(allStationIDList))]
    for i in range(len(allStationIDList)):
        for j in range(len(allStationIDList)):
            if i != j:
                distance = computeDistanceBetweenAB(allStationIDList[i], allStationIDList[j])
                if distance == 0:
                    distance = 100
                graphMatrix[i][j] = 1.0 / distance
    graphMatrix = np.array(graphMatrix, dtype=np.float32)
    for i in range(len(allStationIDList)):
        for j in range(len(allStationIDList)):
            graphMatrix[i][j] = graphMatrix[i][j] / np.sum(graphMatrix[i])
    for i in range(len(allStationIDList)):
        graphMatrix[i][i] = 1
    np.savetxt(os.path.join(txtPath, 'distanceGraphMatrix.txt'), graphMatrix, newline='\n', delimiter=' ')

    # demand graph
    transitionMatrix = np.loadtxt(os.path.join(txtPath, 'transitionMatrix.txt'), delimiter=' ')
    demandGraphMatrix = np.array(transitionMatrix, dtype=np.float32).transpose()
    for i in range(len(allStationIDList)):
        demandSum = np.sum(demandGraphMatrix[i])
        if demandSum == 0:
            demandSum = 1
        for j in range(len(allStationIDList)):
            demandGraphMatrix[i][j] = demandGraphMatrix[i][j] / demandSum
    for i in range(len(allStationIDList)):
        demandGraphMatrix[i][i] = demandGraphMatrix[i][i] + 1
    np.savetxt(os.path.join(txtPath, 'demandGraphMatrix.txt'), demandGraphMatrix, newline='\n', delimiter=' ')

    # demand mask
    demandMask = np.array([[0 for _ in range(len(allStationIDList))] for _ in range(len(centralStationIDList))])
    for i in range(len(centralStationIDList)):
        for j in range(len(allStationIDList)):
            if centralStationIDList[i] == allStationIDList[j]:
                demandMask[i][j] = 1
    np.savetxt(os.path.join(txtPath, 'demandMask.txt'), demandMask, newline='\n', delimiter=' ')

    # fusion graph
    # demand / distance
    fusionGraphMatrix = [[0 for _ in range(len(allStationIDList))] for _ in range(len(allStationIDList))]
    inDemandTransition = transitionMatrix.transpose()
    for i in range(len(allStationIDList)):
        for j in range(len(allStationIDList)):
            if i != j:
                distance = computeDistanceBetweenAB(allStationIDList[i], allStationIDList[j])
                if distance == 0:
                    distance = 100
                fusionGraphMatrix[i][j] = inDemandTransition[i, j] / distance
    fusionGraphMatrix = np.array(fusionGraphMatrix, dtype=np.float32)
    for i in range(len(allStationIDList)):
        sumRow = np.sum(fusionGraphMatrix[i])
        if sumRow == 0:
            sumRow = 1
        for j in range(len(allStationIDList)):
            fusionGraphMatrix[i][j] = fusionGraphMatrix[i][j] / sumRow
    for i in range(len(allStationIDList)):
        fusionGraphMatrix[i][i] = 1
    np.savetxt(os.path.join(txtPath, 'fusionGraphMatrix.txt'), fusionGraphMatrix, newline='\n', delimiter=' ')

    # fusion graph
    fusionGraphMatrix2 = 0.5 * graphMatrix + 0.5 * demandGraphMatrix
    np.savetxt(os.path.join(txtPath, 'fusionGraphMatrix2.txt'), fusionGraphMatrix2, newline='\n', delimiter=' ')

    # demand correlation graph
    # (1) get the demand list of stations
    if os.path.isfile(os.path.join(txtPath, 'demandListForGraph.txt')) is False:
        demandList = []
        timeRange = ['2016-01-01', '2017-01-01']
        timeSlot = 60
        for stationIndex in range(len(allStationIDList)):
            date = parse(timeRange[0])
            endData = parse(timeRange[1])
            stationMinDemandData = getJsonDataFromPath(os.path.join(demandMinDataPath, allStationIDList[stationIndex] + '.json'))
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
                        sumList.append(max(0, result[0] + result[1]))
                    dayIn.append([sum(inList[e:e + timeSlot]) for e in range(len(inList)) if
                                  e % timeSlot == 0])
                    dayOut.append([sum(outList[e:e + timeSlot]) for e in range(len(outList)) if
                                   e % timeSlot == 0])
                    daySum.append([sum(sumList[e:e + timeSlot]) for e in range(len(sumList)) if
                                   e % timeSlot == 0])
                date = date + datetime.timedelta(days=1)
            demandList.append(reduce(lambda x,y:x+y, dayIn))
        np.savetxt(os.path.join(txtPath, 'demandListForGraph.txt'), demandList, delimiter=' ', newline='\n')
    else:
        demandList = np.loadtxt(os.path.join(txtPath, 'demandListForGraph.txt'), delimiter=' ')

    checkZeroList = [checkZero(demandList[e]) for e in range(len(demandList))]
    correlationGraph = [[0 for _ in range(len(allStationIDList))] for _ in range(len(allStationIDList))]
    for i in range(len(allStationIDList)):
        for j in range(len(allStationIDList)):
            if i == j:
                continue
            if checkZeroList[i] or checkZeroList[j]:
                correlationGraph[i][j] = 0
            else:
                correlationGraph[i][j] = pearsonr(demandList[i], demandList[j])[0]
    for i in range(len(allStationIDList)):
        rowSum = np.sum(correlationGraph[i])
        if rowSum == 0:
            rowSum = 1
        for j in range(len(allStationIDList)):
            correlationGraph[i][j] = correlationGraph[i][j] / rowSum
    for i in range(len(allStationIDList)):
        correlationGraph[i][i] = correlationGraph[i][i] + 1
    np.savetxt(os.path.join(txtPath, 'correlationGraphMatrix.txt'), correlationGraph, newline='\n', delimiter=' ')
