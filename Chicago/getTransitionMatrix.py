from dataAPI.apis import *
from sharedParametersV2 import *

def getTransitionMatrixSlave(timeStringRange, stationIDList, p, myRank):
    print('Threads', myRank)
    start = parse(timeStringRange[0])
    end = parse(timeStringRange[1])
    inTimeRange = lambda x: True if x > start and x < end else False
    csvFileNameList = [e for e in os.listdir(ChicagoCSVDataPath) if e.endswith(".csv") and 'Trips' in e]
    transitionMatrix = [[0 for _ in range(stationIDList.__len__())] for _ in range(stationIDList.__len__())]
    for csvFile in csvFileNameList:
        with open(os.path.join(ChicagoCSVDataPath, csvFile), 'r') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            print(csvFile)
            step = 0
            for row in f_csv:
                if step % p == myRank:
                    # get all the data
                    startTime = parse(row[1])
                    stopTime = parse(row[2])
                    startStationID = row[5]
                    endStationID = row[7]
                    if inTimeRange(startTime) and inTimeRange(stopTime) and startStationID in stationIDList and\
                        endStationID in stationIDList:
                        transitionMatrix[stationIDList.index(startStationID)][stationIDList.index(endStationID)] += 1
                step += 1
    np.savetxt(os.path.join(txtPath, 'transitionMatrix-%s.txt' % myRank), np.array(transitionMatrix, dtype=np.int32), delimiter=' ', newline='\n')


if __name__ == '__main__':
    stationIDList = getStationIDList()

    n_jobs = 12
    timeRangeTransitionMatrix = ['2016-10-01', '2017-01-01']

    # timeRangeTransitionMatrix = trainDataTimeRange

    p = Pool()
    for i in range(n_jobs):
        p.apply_async(getTransitionMatrixSlave, args=(timeRangeTransitionMatrix, stationIDList, n_jobs, i))
    p.close()
    p.join()
    transitionMatrixList = []
    for i in range(n_jobs):
        transitionMatrixList.append(np.loadtxt(os.path.join(txtPath, 'transitionMatrix-%s.txt' % i), delimiter=' '))
        os.remove(os.path.join(txtPath, 'transitionMatrix-%s.txt' % i))
    transitionMatrix = reduce(lambda x, y: x + y, transitionMatrixList)
    np.savetxt(os.path.join(txtPath, 'transitionMatrix-{}.txt'.format('-'.join(timeRangeTransitionMatrix))), np.array(transitionMatrix, dtype=np.int32),
               delimiter=' ', newline='\n')