from multiprocessing import Pool
import os

targetFolder = 'GraphFusionModel'
targetScript = 'GraphFusionModelV14'

# targetFolder = 'ARIMA'
# targetScript = 'ARIMA-V1'

# targetFolder = 'SingleGraph'
# targetScript = 'SingleGraph-V1'

def slaveThread(fileNameString, argv):
    os.system('python -m ' + fileNameString + ' ' + argv)

stationRangeList = [
    [0, 30]
]

if __name__ == '__main__':

    n_jobs = 4
    stationRange = stationRangeList[0]

    k = stationRange[0]
    while k <= stationRange[1]:
        currentJobNumber = min(n_jobs, stationRange[1] - k + 1)
        print('Total process', currentJobNumber)
        p = Pool()
        for i in range(currentJobNumber):
            p.apply_async(slaveThread, args=(targetFolder + '.' + targetScript + '.py', targetScript + '_%s' % (i + k)), )
        p.close()
        p.join()
        k += currentJobNumber