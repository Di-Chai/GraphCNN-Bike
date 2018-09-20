from multiprocessing import Pool
import os

def slaveThread(fileNameString, argv):
    os.system('python ' + fileNameString + ' ' + argv)

fileRangeList = [
    [0, 29],
]

if __name__ == '__main__':

    n_jobs = 6
    fileRange = fileRangeList[0]

    k = fileRange[0]
    while k <= fileRange[1]:
        currentJobNumber = min(n_jobs, fileRange[1] - k + 1)
        print('Total process', currentJobNumber)
        p = Pool()
        for i in range(currentJobNumber):
            p.apply_async(slaveThread, args=('GraphSingleStationDemandPreV6.py', 'GraphSingleStationDemandPreV6_%s' % (i + k)), )
        p.close()
        p.join()
        k += currentJobNumber