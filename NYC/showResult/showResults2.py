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
def drawSingleStation(myRank, n_jobs, savePath, finalPre, testTarget, uncertainty, plotFlag, stationNumber):
    print('***************Child process %s*****************' % os.getpid())
    RMSE = {}
    RMSE[str(stationNumber)] = {}
    dailyRMSE = []
    for dayCounter in range(len(finalPre)):
        if dayCounter % n_jobs == myRank:
            x = range(len(finalPre[dayCounter]))
            y_pre = finalPre[dayCounter]
            y_actual = testTarget[dayCounter]
            error = uncertainty[dayCounter] * Z

            currentDayRMSE = np.mean((y_pre - y_actual)**2)**0.5
            print('day', dayCounter, 'RMSE', currentDayRMSE)
            dailyRMSE.append(currentDayRMSE)

            if plotFlag:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)

                font1 = {'family': 'Times New Roman',
                                          'weight': 'normal',
                                          'size': 60,
                                          }

                ax1.plot(x, y_actual, 'r-', label='Real In-Demand')
                ax1.plot(x, y_pre, 'b--', label='Prediction In-Demand', linewidth=3.0)
                ax1.fill_between(x, y_pre - error, y_pre + error, color=(0.5, 0.5, 0.5, 0.5),
                                 label='Confidence Interval')
                ax1.set_ylabel('In-Demand', font1)
                ax1.set_xlabel('Time(60min)', font1)
                if stationNumber == 2:
                    ax1.set_title('Station near school\nRMSE=%s\nUncertainty' % (currentDayRMSE),
                                  font1)
                elif stationNumber == 5:
                    ax1.set_title('Station near residential area\nRMSE=%s\nUncertainty' % (currentDayRMSE),
                                  font1)
                else:
                    ax1.set_title('Prediction and Real In-Demand\nRMSE=%s\nUncertainty' % (currentDayRMSE),
                              font1)
                from matplotlib.pyplot import gca

                a = gca()
                a.set_xticklabels(a.get_xticks(), font1)
                a.set_yticklabels(a.get_yticks(), font1)
                plt.legend(loc='upper right', prop=font1)
                plt.grid()
                fig.set_size_inches(60, 30)
                if not os.path.exists(os.path.join(savePath, str(stationNumber))):
                    os.makedirs(os.path.join(savePath, str(stationNumber)))
                fig.savefig(os.path.join(os.path.join(savePath, str(stationNumber)), 'in-demand-pre-%s' % dayCounter),
                            dpi=100)
                plt.close()
            RMSE[str(stationNumber)][str(dayCounter)] = currentDayRMSE
    saveJsonData(RMSE, 'dailyRMSE-%s-%s.json' % (stationNumber, myRank))


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
        stationNumber = int(currentCodeVersion.split('-')[-1])
        print(currentCodeVersion)
        savePath = os.path.join(pngPath, currentCodeVersion)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        finalPreResult = np.loadtxt(os.path.join(targetFolder, txtFileName[i]), delimiter=' ').reshape([-1, 1440-featureLength-targetTimeSlot+1, targetLength])
        testTarget = np.sum(np.loadtxt(os.path.join(targetFolder, txtFileName[i+1]), delimiter=' '), axis=1).reshape([-1, 1440-featureLength-targetTimeSlot+1, targetLength])
        uncertainty = np.loadtxt(os.path.join(targetFolder, txtFileName[i+2]), delimiter=' ').reshape([-1, 1440-featureLength-targetTimeSlot+1, targetLength])

        finalPreResult = np.sum(finalPreResult, axis=2)
        testTarget = np.sum(testTarget, axis=2)
        uncertainty = np.mean(uncertainty, axis=2)

        p = Pool()
        for rank in range(n_jobs):
            p.apply_async(drawSingleStation,
                          args=(rank, n_jobs, savePath, finalPreResult, testTarget, uncertainty, True, stationNumber), )
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
    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        singleStationRMSE = [e[1] for e in sorted(RMSE[currentCodeVersion].items(), key=lambda x:int(x[0]), reverse=False)]
        RMSEList.append(singleStationRMSE)
        print(i, np.mean(singleStationRMSE))
        if i >= 1:
            TTest = stats.ttest_ind(RMSEList[int(i/fileGap)], RMSEList[int((i-1)/fileGap)], equal_var=False)
            ttest = TTest[0]
            pValue = TTest[1]
            # print(i/fileGap, ttest, pValue)