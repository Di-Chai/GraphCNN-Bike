import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataAPI.utils import *
from scipy import stats
from sharedParametersV2 import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from functools import reduce

def checkZero(valueList):
    for e in valueList:
        if e != 0:
            return False
    return True

def flatList(valueList):
    resultList = valueList[0]
    for i in range(1, valueList.__len__()):
        resultList = resultList + valueList[i]
    return resultList

Z = 1.96  # a = 95%
def drawSingleStation(myRank, n_jobs, savePath, finalPre, testTarget, plotFlag, stationNumber):
    print('***************Child process %s*****************' % os.getpid())
    RMSE = {}
    RMSE[str(stationNumber)] = {}
    dailyRMSE = []
    for dayCounter in range(len(finalPre)):
        if dayCounter % n_jobs == myRank:
            x = range(len(finalPre[dayCounter])-6)
            y_pre = finalPre[dayCounter][6:]
            y_actual = testTarget[dayCounter][6:]

            if checkZero(y_actual):
                currentDayRMSE = 0
            else:
                currentDayRMSE = np.mean((y_pre - y_actual)**2)**0.5

            print('day', dayCounter, 'RMSE', currentDayRMSE)
            dailyRMSE.append(currentDayRMSE)

            if plotFlag:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(x, y_actual, 'r-', label='Real In-Demand')
                ax1.plot(x, y_pre, 'b--', label='Prediction In-Demand', linewidth=3.0)
                ax1.set_ylabel('In-Demand', fontsize=40)
                ax1.set_xlabel('Time(60min)', fontsize=40)
                ax1.set_title('Prediction and Real In-Demand\nRMSE=%s\nUncertainty' % (currentDayRMSE), fontsize=50)
                plt.xticks(fontsize=40)
                plt.yticks(fontsize=40)
                plt.legend(loc='upper right', fontsize=40)
                plt.grid()
                fig.set_size_inches(60, 30)
                fig.savefig(os.path.join(savePath, 'in-demand-pre-%s' % dayCounter),
                            dpi=100)
                plt.close()
            RMSE[str(stationNumber)][str(dayCounter)] = currentDayRMSE
    saveJsonData(RMSE, 'dailyRMSE-%s-%s.json' % (stationNumber, myRank))


if __name__ == '__main__':
    codeVersion = 'GraphSingleStationDemandPreV8'
    targetFolder = GraphDemandPreDataPath
    txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt') and e.startswith(codeVersion)]
    txtFileRankList = [int(e.split('-')[1]) for e in txtFileName]
    txtFileName = [e[1] for e in sorted(zip(txtFileRankList, txtFileName), key = lambda x: x[0])]
    print(txtFileName)
    fileGap = 2
    RMSE = {}
    n_jobs = 12

    plotLength = 14

    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(currentCodeVersion.split('-')[-1])
        print(currentCodeVersion)
        savePath = os.path.join(pngPath, currentCodeVersion)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        finalPreResult = np.loadtxt(os.path.join(targetFolder, txtFileName[i]), delimiter=' ').reshape([-1, 24])[0:plotLength, :]
        testTarget = np.loadtxt(os.path.join(targetFolder, txtFileName[i+1]), delimiter=' ').reshape([-1, 24])[0:plotLength, :]

        p = Pool()
        for rank in range(n_jobs):
            p.apply_async(drawSingleStation,
                          args=(rank, n_jobs, savePath, finalPreResult, testTarget, True, stationNumber), )
        p.close()
        p.join()

        currentVersionRMSE = {}
        for rank in range(n_jobs):
            partRMSE = getJsonData('dailyRMSE-%s-%s.json' % (stationNumber, rank))
            for key,value in partRMSE[str(stationNumber)].items():
                currentVersionRMSE[key] = value
            removeJsonData('dailyRMSE-%s-%s.json' % (stationNumber, rank))
        RMSE[currentCodeVersion] = currentVersionRMSE
    saveJsonData(RMSE, 'RMSE-GraphPre.json')

    RMSEList = []
    singleStationRMSEList = []
    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(currentCodeVersion.split('-')[-1])
        singleStationRMSE = [e[1] for e in sorted(RMSE[currentCodeVersion].items(), key=lambda x:int(x[0]), reverse=False)]
        RMSEList.append(singleStationRMSE)
        modelName = 'ARIMA'
        if np.mean(singleStationRMSE) == 0:
            continue
        print(stationNumber, modelName, np.mean(singleStationRMSE))
        singleStationRMSEList.append(np.mean(singleStationRMSE))

    print('Total Station Number', singleStationRMSEList.__len__())
    print('AllMean:', np.mean(singleStationRMSEList))

    np.savetxt(os.path.join(txtPath, codeVersion + '_%s' % plotLength + '.txt'), np.array(singleStationRMSEList, dtype=np.float32), delimiter=' ',
               newline='\n')

