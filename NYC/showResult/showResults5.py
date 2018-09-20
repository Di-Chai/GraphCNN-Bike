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
            x = [e+6 for e in range(len(finalPre[dayCounter]))]
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

                font1 = {'family': 'Times New Roman',
                         'weight': 'normal',
                         'size': 100,
                         }

                # ax1.plot(x, y_actual, 'r-', label='Real In-Demand')
                ax1.plot(x, y_pre, 'b--', label='Prediction inflow', linewidth=3.0)
                ax1.fill_between(x,
                                 [e if e > 0 else 0 for e in y_pre - error],
                                 y_pre + error, color=(0.5, 0.5, 0.5, 0.5),
                                 label='Confidence Interval')
                ax1.set_ylabel('Inflow', font1)
                ax1.set_xlabel('Time (Hour)', font1)

                # ax1.set_title('Prediction and Real Check In\nRMSE=%.4f\nConfidence=%.4f' % (
                # currentDayRMSE, np.mean(uncertaintyList)), fontsize=70)

                my_x_ticks = np.arange(5, 24, 2)  # 显示范围为-5至5，每0.5显示一刻度
                my_y_ticks = np.arange(0, 70, 10)  # 显示范围为-2至2，每0.2显示一刻度
                plt.xticks(my_x_ticks)
                plt.yticks(my_y_ticks)

                plt.xlim((5, 24))
                plt.ylim((0, 60))

                from matplotlib.pyplot import gca

                a = gca()
                a.set_xticklabels(a.get_xticks(), font1)
                a.set_yticklabels(a.get_yticks(), font1)
                plt.legend(loc='upper right', prop=font1)

                plt.grid()

                fig.set_size_inches(60, 40)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                fig.savefig(os.path.join(savePath, 'in-demand-pre-%s' % dayCounter),
                            dpi=50)
                plt.close()
            RMSE[str(stationNumber)][str(dayCounter)] = currentDayRMSE
            Uncertainty[str(stationNumber)][str(dayCounter)] = np.mean(uncertaintyList)
    saveJsonData(RMSE, 'dailyRMSE-%s-%s-%s.json' % (stationNumber, myRank, codeVersion))
    saveJsonData(Uncertainty, 'dailyUncertainty-%s-%s-%s.json' % (stationNumber, myRank, codeVersion))


if __name__ == '__main__':
    codeVersion = 'GraphFusionModelV10'
    targetFolder = GraphDemandPreDataPath
    txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt') and e.startswith(codeVersion)]
    txtFileRankList = [int(e.split('-')[1]) for e in txtFileName]
    txtFileName = [e[1] for e in sorted(zip(txtFileRankList, txtFileName), key = lambda x: x[0])]
    print(txtFileName)

    fileGap = 3
    RMSE = {}
    Uncertainty = {}
    n_jobs = 10

    plotDayLength = 80

    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(int(currentCodeVersion.split('-')[-1])/3)
        print(currentCodeVersion)
        savePath = os.path.join(pngPath, currentCodeVersion)

        finalPreResult = np.loadtxt(os.path.join(targetFolder, txtFileName[i]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotDayLength, :]
        testTarget = np.loadtxt(os.path.join(targetFolder, txtFileName[i+1]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotDayLength, :]
        uncertainty = np.loadtxt(os.path.join(targetFolder, txtFileName[i+2]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotDayLength, :]

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