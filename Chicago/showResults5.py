import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataAPI.apis import *
from scipy import stats
from sharedParametersV2 import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from functools import reduce

def flatList(valueList):
    resultList = valueList[0]
    for i in range(1, valueList.__len__()):
        resultList = resultList + valueList[i]
    return resultList

def checkZero(valueList):
    for e in valueList:
        if e != 0:
            return False
    return True

Z = 1.92  # a = 95%
def drawSingleStation(myRank, n_jobs, savePath, finalPre, testTarget, uncertainty, plotFlag, stationNumber, codeVersion):
    print('***************Child process %s*****************' % os.getpid())
    RMSE = {}
    Uncertainty = {}
    RMSE[str(stationNumber)] = {}
    Uncertainty[str(stationNumber)] = {}
    dailyRMSE = []
    for dayCounter in range(len(finalPre)):
        if dayCounter % n_jobs == myRank:
            x = range(len(finalPre[dayCounter]))
            y_pre = finalPre[dayCounter]
            y_actual = testTarget[dayCounter]
            error = uncertainty[dayCounter] * Z

            if checkZero(y_actual):
                currentDayRMSE = 0
            else:
                currentDayRMSE = np.mean((y_pre - y_actual)**2)**0.5

            print('day', dayCounter, 'RMSE', currentDayRMSE)
            dailyRMSE.append(currentDayRMSE)

            uncertaintyList = [1.0 if y_actual[e] >= (y_pre-error)[e] and y_actual[e] <= (y_pre+error)[e] else 0.0 for e in range(len(y_actual))]

            if plotFlag:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(x, y_actual, 'r-', label='Real Check In')
                ax1.plot(x, y_pre, 'b--', label='Predicted Check In', linewidth=3.0)
                ax1.fill_between(x, y_pre - error, y_pre + error, color=(0.5, 0.5, 0.5, 0.5),
                                 label='Confidence Interval')
                ax1.set_ylabel('Check In Amount', fontsize=60)
                ax1.set_xlabel('Time(Hour)', fontsize=60)

                ax1.set_title('Prediction and Real Check In\nRMSE=%.4f\nConfidence=%.4f' % (currentDayRMSE, np.mean(uncertaintyList)), fontsize=70)
                plt.xticks(fontsize=60)
                plt.yticks(fontsize=60)
                plt.legend(loc='upper right', fontsize=60)
                plt.grid()
                fig.set_size_inches(60, 30)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                fig.savefig(os.path.join(savePath, 'in-demand-pre-%s' % dayCounter),
                            dpi=100)
                plt.close()
            RMSE[str(stationNumber)][str(dayCounter)] = currentDayRMSE
            Uncertainty[str(stationNumber)][str(dayCounter)] = np.mean(uncertaintyList)
    saveJsonData(RMSE, 'dailyRMSE-%s-%s-%s.json' % (stationNumber, myRank, codeVersion))
    saveJsonData(Uncertainty, 'dailyUncertainty-%s-%s-%s.json' % (stationNumber, myRank, codeVersion))


if __name__ == '__main__':
    codeVersion = 'GraphFusionModelV12'
    targetFolder = GraphDemandPreDataPath
    txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt') and e.startswith(codeVersion)]
    txtFileRankList = [int(e.split('-')[1]) for e in txtFileName]
    txtFileName = [e[1] for e in sorted(zip(txtFileRankList, txtFileName), key = lambda x: x[0])]
    print(txtFileName)

    fileGap = 3
    RMSE = {}
    Uncertainty = {}
    n_jobs = 6

    plotDayLength = 14

    # featureLength = 12

    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(int(currentCodeVersion.split('-')[-1])/3)
        print(currentCodeVersion)
        savePath = os.path.join(pngPath, currentCodeVersion)

        finalPreResult = np.loadtxt(os.path.join(targetFolder, txtFileName[i]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotDayLength, :]
        testTarget = np.loadtxt(os.path.join(targetFolder, txtFileName[i+1]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotDayLength, :]
        uncertainty = np.row_stack((np.loadtxt(os.path.join(targetFolder, txtFileName[i+2]), delimiter=' ')
                                    for _ in range(1440))).reshape([-1, 24-featureLength-targetLength+1])[:plotDayLength, :]

        p = Pool()
        for rank in range(n_jobs):
            p.apply_async(drawSingleStation,
                          args=(rank, n_jobs, savePath, finalPreResult, testTarget, uncertainty, True, stationNumber, codeVersion), )
        p.close()
        p.join()

        currentVersionRMSE = {}
        currentVersionUncertainty = {}
        for rank in range(n_jobs):
            partRMSE = getJsonData('dailyRMSE-%s-%s-%s.json' % (stationNumber, rank, codeVersion))
            partUncertainty = getJsonData('dailyUncertainty-%s-%s-%s.json' % (stationNumber, rank, codeVersion))
            for key, value in partRMSE[str(stationNumber)].items():
                currentVersionRMSE[key] = value
            for key, value in partUncertainty[str(stationNumber)].items():
                currentVersionUncertainty[key] = value
            removeJsonData('dailyRMSE-%s-%s-%s.json' % (stationNumber, rank, codeVersion))
            removeJsonData('dailyUncertainty-%s-%s-%s.json' % (stationNumber, rank, codeVersion))
        RMSE[currentCodeVersion] = currentVersionRMSE
        Uncertainty[currentCodeVersion] = currentVersionUncertainty
    saveJsonData(RMSE, 'RMSE-GraphPre.json')

    RMSEList = []
    singleStationRMSEList = []
    singleStationUncertaintyList = []
    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(currentCodeVersion.split('-')[-1])

        singleStationRMSE = [e[1] for e in sorted(RMSE[currentCodeVersion].items(), key=lambda x:int(x[0]), reverse=False)]

        singleStationUncertainty = [[e[1] for e in sorted(Uncertainty[currentCodeVersion].items(), key=lambda x:int(x[0]), reverse=False)]]

        RMSEList.append(singleStationRMSE)
        modelName = 'FusionModel'
        print(stationNumber, modelName, np.mean(singleStationRMSE), np.mean(singleStationUncertainty))
        if np.mean(singleStationRMSE) == 0:
            continue
        singleStationRMSEList.append(np.mean(singleStationRMSE))
        singleStationUncertaintyList.append(singleStationUncertainty)

    print('Total Station Number', singleStationRMSEList.__len__())
    print('AllMean:', np.mean(singleStationRMSEList), np.mean(singleStationUncertaintyList))

    np.savetxt(os.path.join(txtPath, codeVersion + '_%s' % plotDayLength + '.txt'), np.array(singleStationRMSEList, dtype=np.float32),
               delimiter=' ', newline='\n')