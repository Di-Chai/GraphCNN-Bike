import numpy as np
from localPath import *
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataAPI.utils import *
from scipy import stats
from sharedParameters import *

def flatList(valueList):
    resultList = valueList[0]
    for i in range(1, valueList.__len__()):
        resultList = resultList + valueList[i]
    return resultList

Z = 1.96  # a = 95%
def drawSingleStation(myRank, n_jobs, savePath, finalPre, testTarget, uncertainty, plotFlag):
    print('***************Child process %s*****************' % os.getpid())
    RMSE = {}
    for stationCounter in range(len(finalPre)):
        if stationCounter % n_jobs == myRank:
            dailyRMSE = []
            for dayCounter in range(len(finalPre[stationCounter])):
                print('station', stationCounter, 'day', dayCounter)
                x = range(len(finalPre[stationCounter][dayCounter]))
                y_pre = finalPre[stationCounter][dayCounter]
                y_actual = testTarget[stationCounter][dayCounter]
                error = uncertainty[stationCounter][dayCounter] * Z

                dailyRMSE.append(np.mean((y_pre - y_actual)**2)**0.5)

                if plotFlag:
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.plot(x, y_actual, 'r-', label='Real In-Demand')
                    ax1.plot(x, y_pre, 'b--', label='Prediction In-Demand', linewidth=3.0)
                    ax1.fill_between(x, y_pre - error, y_pre + error, color=(0.5, 0.5, 0.5, 0.5),
                                     label='Confidence Interval')
                    ax1.set_ylabel('In-Demand', fontsize=40)
                    ax1.set_xlabel('Time(60min)', fontsize=40)

                    ax1.set_title('Prediction and Real In-Demand\nRMSE=%s\nUncertainty:%s', fontsize=50)
                    plt.xticks(fontsize=40)
                    plt.yticks(fontsize=40)
                    plt.legend(loc='upper right', fontsize=40)
                    plt.grid()
                    fig.set_size_inches(60, 30)
                    if not os.path.exists(os.path.join(savePath, str(stationCounter))):
                        os.makedirs(os.path.join(savePath, str(stationCounter)))
                    fig.savefig(os.path.join(os.path.join(savePath, str(stationCounter)), 'in-demand-pre-%s' % dayCounter),
                                dpi=100)
                    plt.close()
                RMSE[stationCounter] = dailyRMSE
    saveJsonData(RMSE, 'dailyRMSE-%s.json' % myRank)


if __name__ == '__main__':
    targetFolder = GraphDemandPreDataPath
    txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt')]
    txtFileName = sorted(txtFileName, key = lambda x: x)
    print(txtFileName)

    fileGap = 3
    RMSE = {}
    n_jobs = 12

    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        print(currentCodeVersion)
        savePath = os.path.join(pngPath, currentCodeVersion)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        finalPreResult = np.loadtxt(os.path.join(targetFolder, txtFileName[i]), delimiter=' ').reshape([-1, 1440-featureLength-targetLength+1])
        testTarget = np.loadtxt(os.path.join(targetFolder, txtFileName[i+1]), delimiter=' ').reshape([-1, 1440-featureLength-targetLength+1])
        uncertainty = np.loadtxt(os.path.join(targetFolder, txtFileName[i+2]), delimiter=' ').reshape([-1, 1440-featureLength-targetLength+1])

        p = Pool()
        for i in range(n_jobs):
            p.apply_async(drawSingleStation,
                          args=(i, n_jobs, savePath, finalPreResult, testTarget, uncertainty, False), )
        p.close()
        p.join()

        currentVersionRMSE = {}
        for i in range(n_jobs):
            partRMSE = getJsonData('dailyRMSE-%s.json' % i)
            for key,value in partRMSE.items():
                currentVersionRMSE[key] = value
            removeJsonData('dailyRMSE-%s.json' % i)
        RMSE[currentCodeVersion] = currentVersionRMSE
    saveJsonData(RMSE, 'RMSE-GraphPre.json')

    RMSEList = []
    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        singleStationRMSE = np.mean([e[1] for e in sorted(RMSE[currentCodeVersion].items(), key=lambda x:int(x[0]), reverse=False)], axis=1)
        RMSEList.append(singleStationRMSE)
        print(i, np.mean(singleStationRMSE))
        if i >= 1:
            TTest = stats.ttest_ind(RMSEList[int(i/fileGap)], RMSEList[int((i-1)/fileGap)], equal_var=False)
            ttest = TTest[0]
            pValue = TTest[1]
            # print(i/fileGap, ttest, pValue)