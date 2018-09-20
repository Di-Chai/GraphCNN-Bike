import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataAPI.utils import *
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
def drawSingleStation(myRank, n_jobs, savePath, finalPre, testTarget, uncertainty, plotFlag, stationNumber):
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
                ax1.plot(x, y_pre, 'b--', label='Predicted inflow', linewidth=3.0)
                ax1.fill_between(x, y_pre - error, y_pre + error, color=(0.5, 0.5, 0.5, 0.5),
                                 label='Confidence Interval')

                font1 = {'family': 'Times New Roman',
                         'weight': 'normal',
                         'size': 120,
                         }

                ax1.set_ylabel('Inflow', font1)
                ax1.set_xlabel('Time(Hour)', font1)

                if stationNumber == 2:
                    ax1.set_title('Station near school\nRMSE=%.4f\nConfidence=%.4f' % (currentDayRMSE, np.mean(uncertaintyList)), font1)
                elif stationNumber == 5:
                    ax1.set_title('Station near residential area\nRMSE=%.4f\nConfidence=%.4f' % (
                    currentDayRMSE, np.mean(uncertaintyList)), font1)
                else:
                    ax1.set_title('Prediction and Real Check In\nRMSE=%.4f\nConfidence=%.4f' % (
                    currentDayRMSE, np.mean(uncertaintyList)), font1)

                from matplotlib.pyplot import gca

                a = gca()
                a.set_xticklabels(a.get_xticks(), font1)
                a.set_yticklabels(a.get_yticks(), font1)
                plt.legend(loc='upper right', prop=font1)
                plt.grid()
                fig.set_size_inches(60, 40)
                fig.savefig(os.path.join(savePath, 'in-demand-pre-%s' % dayCounter),
                            dpi=50)
                plt.close()
            RMSE[str(stationNumber)][str(dayCounter)] = currentDayRMSE
            Uncertainty[str(stationNumber)][str(dayCounter)] = np.mean(uncertaintyList)
    saveJsonData(RMSE, 'dailyRMSE-%s-%s.json' % (stationNumber, myRank))
    saveJsonData(Uncertainty, 'dailyUncertainty-%s-%s.json' % (stationNumber, myRank))


if __name__ == '__main__':
    codeVersion = 'GraphSingleStationDemandPreV3'
    targetFolder = GraphDemandPreDataPath
    # targetFolder = "D:\\V2OrderByDegree"
    txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt') and e.startswith(codeVersion)]
    txtFileRankList = [int(e.split('-')[1]) for e in txtFileName]
    txtFileName = [e[1] for e in sorted(zip(txtFileRankList, txtFileName), key = lambda x: x[0])]
    print(txtFileName)

    fileGap = 3
    RMSE = {}
    Uncertainty = {}
    n_jobs = 10

    plotLength = 80

    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(int(currentCodeVersion.split('-')[-1])/3)
        print(currentCodeVersion)
        savePath = os.path.join(pngPath, currentCodeVersion)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        finalPreResult = np.loadtxt(os.path.join(targetFolder, txtFileName[i]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotLength, :]
        testTarget = np.loadtxt(os.path.join(targetFolder, txtFileName[i+1]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotLength, :]
        uncertainty = np.loadtxt(os.path.join(targetFolder, txtFileName[i+2]), delimiter=' ').reshape([-1, 24-featureLength-targetLength+1])[:plotLength, :]

        p = Pool()
        for rank in range(n_jobs):
            p.apply_async(drawSingleStation,
                          args=(rank, n_jobs, savePath, finalPreResult, testTarget, uncertainty, True, stationNumber), )
        p.close()
        p.join()

        currentVersionRMSE = {}
        currentVersionUncertainty = {}
        for rank in range(n_jobs):
            partRMSE = getJsonData('dailyRMSE-%s-%s.json' % (stationNumber, rank))
            partUncertainty = getJsonData('dailyUncertainty-%s-%s.json' % (stationNumber, rank))
            for key,value in partRMSE[str(stationNumber)].items():
                currentVersionRMSE[key] = value
            for key,value in partUncertainty[str(stationNumber)].items():
                currentVersionUncertainty[key] = value
            removeJsonData('dailyRMSE-%s-%s.json' % (stationNumber, rank))
            removeJsonData('dailyUncertainty-%s-%s.json' % (stationNumber, rank))
        RMSE[currentCodeVersion] = currentVersionRMSE
        Uncertainty[currentCodeVersion] = currentVersionUncertainty
    saveJsonData(RMSE, 'RMSE-GraphPre.json')

    RMSEList = []
    UncertaintyList = []
    singleStationRMSEList = []
    singleStationUncertaintyList = []
    for i in range(0, txtFileName.__len__(), fileGap):
        currentCodeVersion = '-'.join(txtFileName[i].split('-')[0:2])
        stationNumber = int(int(currentCodeVersion.split('-')[-1]) / 3)
        singleStationRMSE = [e[1] for e in sorted(RMSE[currentCodeVersion].items(), key=lambda x:int(x[0]), reverse=False)]
        RMSEList.append(singleStationRMSE)
        singleStationUncertainty = [e[1] for e in sorted(Uncertainty[currentCodeVersion].items(), key=lambda x: int(x[0]), reverse=False)]
        UncertaintyList.append(singleStationUncertainty)

        # if stationNumber >= 6:
        #     continue
        
        if int(i/fileGap) % 3 == 0:
            modelName = 'LSTM'
        elif int(i/fileGap) % 3 == 1:
            modelName = 'LSTM+DistanceGraph'
        elif int(i/fileGap) % 3 == 2:
            modelName = 'LSTM+DemandGraph'
        print(stationNumber, modelName, np.mean(singleStationRMSE), np.mean(singleStationUncertainty))

        if np.mean(singleStationRMSE) == 0:
            continue

        singleStationRMSEList.append(np.mean(singleStationRMSE))
        singleStationUncertaintyList.append(np.mean(singleStationUncertainty))

    # LSTM
    r1 = [singleStationRMSEList[i] for i in range(int(singleStationRMSEList.__len__())) if i % fileGap == 0]
    # LSTM + Distance Graph
    r2 = [singleStationRMSEList[i] for i in range(int(singleStationRMSEList.__len__())) if i % fileGap == 1]
    # LSTM + Demand Graph
    r3 = [singleStationRMSEList[i] for i in range(int(singleStationRMSEList.__len__())) if i % fileGap == 2]

    u1 = [singleStationUncertaintyList[i] for i in range(int(singleStationUncertaintyList.__len__())) if i % fileGap == 0]

    u2 = [singleStationUncertaintyList[i] for i in range(int(singleStationUncertaintyList.__len__())) if i % fileGap == 1]

    u3 = [singleStationUncertaintyList[i] for i in range(int(singleStationUncertaintyList.__len__())) if i % fileGap == 2]

    minLength = reduce(lambda x,y: min(x,y), [len(r1), len(r2), len(r3)])
    r1 = r1[0:minLength]
    r2 = r2[0:minLength]
    r3 = r3[0:minLength]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    X = np.arange(minLength) + 1

    ax1.bar(X, r1, alpha=0.9, width=0.2, facecolor='blue', edgecolor='white', label='LSTM', lw=1)
    ax1.bar(X + 0.2, r2, alpha=0.9, width=0.2, facecolor='red', edgecolor='white', label='LSTM+distanceGraph', lw=1)
    ax1.bar(X + 0.4, r3, alpha=0.9, width=0.2, facecolor='y', edgecolor='white', label='LSTM+demandGraph', lw=1)
    ax1.set_xlabel('Station', fontsize=40)
    ax1.set_ylabel('RMSE', fontsize=40)
    ax1.set_title('Experiment on different stations', fontsize=40)

    xmajorLocator = MultipleLocator(1)
    ax1.xaxis.set_major_locator(xmajorLocator)

    plt.legend(loc="upper right")
    fig.set_size_inches(60, 30)
    fig.savefig(os.path.join(pngPath, 'RMSE_Stations-%s.jpg' % codeVersion), dpi=50)
    plt.close()

    r1_mean = np.mean(r1)
    r2_mean = np.mean(r2)
    r3_mean = np.mean(r3)
    u1_mean = np.mean(u1)
    u2_mean = np.mean(u2)
    u3_mean = np.mean(u3)
    allMeanRMSE = np.mean([r1_mean, r2_mean, r3_mean])
    graphMean = np.mean([r2_mean, r3_mean])

    print('Total Station Number', minLength)
    print('LSTM:', r1_mean, u1_mean, '\n',
          'LSTM+DistanecGraph:', r2_mean, u2_mean, '\n',
          'LSTM+DemandGraph:', r3_mean, u3_mean, '\n',
          'GraphMean', graphMean, np.mean([u2_mean, u3_mean]), '\n',
          'AllMean:', allMeanRMSE, np.mean([u1_mean, u2_mean, u3_mean]),)

    np.savetxt(os.path.join(txtPath, codeVersion + '_%s' % plotLength + '.txt'), np.array([r1, r2, r3, u1, u2, u3], dtype=np.float32), delimiter=' ', newline='\n')
