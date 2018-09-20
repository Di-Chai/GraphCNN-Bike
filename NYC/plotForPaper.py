import numpy as np
import matplotlib.pyplot as plt
from localPath import *
from functools import reduce
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataAPI.utils import *
from sharedParametersV2 import *
import random

colorList = [
    [165, 0, 0],
    [255, 0, 0],
    [255, 165, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 127, 255],
    [0, 0, 255],
    [139, 0, 255]
]

# paperPath = 'C:\\Users\\Elsa\\Dropbox\\PaperWriting\\LatexPaper'
paperPath = pngPath

def autolabel(rects, fontSize=15):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2 - 0.2, height + 0.01, '%.4f' % float(height), fontsize=fontSize)


if __name__ == '__main__':
    imgSize = (8, 10)
    # load data
    singleGraphResult3 = np.loadtxt(os.path.join(txtPath, 'GraphSingleStationDemandPreV3_14.txt'), delimiter=' ')
    result2 = singleGraphResult3[0:3]
    result2u = singleGraphResult3[3:]
    # corrrelation graph
    result3 = np.loadtxt(os.path.join(txtPath, 'GraphSingleStationDemandPreV7_14.txt'), delimiter=' ')
    # fusion graph
    result4 = np.loadtxt(os.path.join(txtPath, 'GraphSingleStationDemandPreV8_14.txt'), delimiter=' ')

    resultFusion = np.loadtxt(os.path.join(txtPath, 'GraphFusionModelV10_14.txt'), delimiter=' ')

    # figure 1
    # top = 30
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # rect = ax1.bar([1, 2, 3, 4], [np.mean(result4[0: top])] + [e for e in np.mean(result2[:, 0:top], axis=1)],
    #                width=0.5)
    # autolabel(rect)
    # ax1.set_ylabel('RMSE', fontsize=20)
    # ax1.set_xlabel('Models', fontsize=20)
    # ax1.set_title('Result on %s stations' % top, fontsize=20)
    # # plt.legend(loc='upper right', fontsize=40)
    # # plt.grid()
    # plt.yticks(fontsize=15)
    # plt.xticks([1, 2, 3, 4], ['SARIMA', 'DeepLSTM', 'DistanceGraph', 'DemandGraph'], fontsize=15)
    # fig.set_size_inches(imgSize[0], imgSize[1])
    # fig.savefig(os.path.join(paperPath, 'result1_%s.jpg' % top), dpi=120)
    # plt.close()
    #
    # # figure 2
    # top = 5
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # rect = ax1.bar([1, 2, 3, 4], [np.mean(result4[0: top])] + [e for e in np.mean(result2[:, 0:top], axis=1)], width=0.5)
    # autolabel(rect)
    # ax1.set_ylabel('RMSE', fontsize=20)
    # ax1.set_xlabel('Models', fontsize=20)
    # ax1.set_title('Result on %s stations' % top, fontsize=20)
    # # plt.legend(loc='upper right', fontsize=40)
    # # plt.grid()
    # plt.yticks(fontsize=15)
    # plt.xticks([1, 2, 3, 4], ['SARIMA', 'DeepLSTM', 'DistanceGraph', 'DemandGraph'], fontsize=15)
    # fig.set_size_inches(imgSize[0], imgSize[1])
    # fig.savefig(os.path.join(paperPath, 'result2_%s.jpg' % top), dpi=120)
    # plt.close()
    #
    # # figure 3
    # top = 10
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # rect = ax1.bar([1, 2, 3, 4], [np.mean(result4[0: top])] + [e for e in np.mean(result2[:, 0:top], axis=1)],
    #                width=0.5)
    # autolabel(rect)
    # ax1.set_ylabel('RMSE', fontsize=20)
    # ax1.set_xlabel('Models', fontsize=20)
    # ax1.set_title('Result on %s stations' % top, fontsize=20)
    # # plt.legend(loc='upper right', fontsize=40)
    # # plt.grid()
    # plt.yticks(fontsize=15)
    # plt.xticks([1, 2, 3, 4], ['SARIMA', 'DeepLSTM', 'DistanceGraph', 'DemandGraph'], fontsize=15)
    # fig.set_size_inches(imgSize[0], imgSize[1])
    # fig.savefig(os.path.jo in(paperPath, 'result3_%s.jpg' % top), dpi=120)
    # plt.close()

    plotLength = 30

    arima = result4[:plotLength]

    r1 = result2[0][:plotLength]
    r2 = result2[1][:plotLength]
    r3 = result2[2][:plotLength]

    corrGraphResult = result3[:plotLength]

    fusionResult = resultFusion[:plotLength]

    minLength = reduce(lambda x, y: min(x, y), [len(r1), len(r2), len(r3)])
    r1 = r1[0:minLength]
    r2 = r2[0:minLength]
    r3 = r3[0:minLength]

    r1_mean = np.mean(r1)
    r2_mean = np.mean(r2)
    r3_mean = np.mean(r3)

    print('Total Station Number', minLength)
    print('ARIMA', np.mean(arima), '\n',
        'LSTM:', r1_mean, '\n',
          'LSTM+DistanecGraph:', r2_mean, '\n',
          'LSTM+DemandGraph:', r3_mean, '\n',
          'LSTM+CorrGraph', np.mean(corrGraphResult), '\n',
          'LSTM+FusionGraph:', np.mean(fusionResult), '\n',)

    print('ARIMA', arima, '\n',
          'LSTM:', r1, '\n',
          'LSTM+DistanecGraph:', r2, '\n',
          'LSTM+DemandGraph:', r3, '\n',
          'LSTM+CorrGraph', corrGraphResult, '\n',
          'LSTM+FusionGraph:', fusionResult, '\n', )

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    X = np.arange(minLength) + 1

    width = 0.12

    ax1.bar(X + width * 0, arima, alpha=0.5, width=width, facecolor='green', edgecolor='white', label='SARIMA', lw=1)
    ax1.bar(X + width * 1, r1, alpha=0.5, width=width, facecolor='blue', edgecolor='white', label='DeepLSTM', lw=1)
    ax1.bar(X + width * 2, r2, alpha=0.5, width=width, facecolor='red', edgecolor='white', label='DistanceGraph', lw=1)
    ax1.bar(X + width * 3, r3, alpha=0.5, width=width, facecolor='y', edgecolor='white', label='RideRecordGraph', lw=1)
    ax1.bar(X + width * 4, corrGraphResult, alpha=0.5, width=width, facecolor='purple', edgecolor='white',
            label='CorrelationGraph', lw=1)
    ax1.bar(X + width * 5, fusionResult, alpha=0.5, width=width, facecolor='orange', edgecolor='white',
            label='FusionGraph', lw=1)

    plt.xticks([], [])
    plt.yticks(fontsize=60)
    ax1.set_xlabel('%s Stations' % plotLength, fontsize=60)
    ax1.set_ylabel('RMSE', fontsize=60)
    # ax1.set_title('Result comparison of NYC', fontsize=40)

    xmajorLocator = MultipleLocator(1)
    ax1.xaxis.set_major_locator(xmajorLocator)

    ax1.grid()

    plt.legend(loc="upper right", fontsize=30)
    fig.set_size_inches(30, 15)
    fig.savefig(os.path.join(pngPath, 'RMSE_Stations_NYC.jpg'), dpi=50)
    plt.close()

    distanceGraphMatrix = np.loadtxt(os.path.join(txtPath, 'distanceGraphMatrix.txt'), delimiter=' ')
    demandGraphMatrix = np.loadtxt(os.path.join(txtPath, 'demandGraphMatrix.txt'), delimiter=' ')
    demandMask = np.loadtxt(os.path.join(txtPath, 'demandMask.txt'), delimiter=' ')

    stationIDDict = getJsonData('centralStationIDList.json')
    centralStationIDList = stationIDDict['centralStationIDList']
    allStationIDList = stationIDDict['allStationIDList']

    ######################################################################################################################
    # Prepare the training data and test data
    ######################################################################################################################
    if os.path.isfile(os.path.join(jsonPath, 'GraphPreData.json')) is False:
        os.system('python getGraphPreDataMulThreads.py')

    GraphValueData = getJsonData('GraphValueMatrix.json')

    GraphValueMatrix = GraphValueData['GraphValueMatrix']

    trainDataLength = len(GraphValueMatrix) - valDataLength - testDataLength

    allTrainData = np.array(GraphValueMatrix[0: trainDataLength + valDataLength], dtype=np.float32)

    allTestData = np.array(GraphValueMatrix[-testDataLength:], dtype=np.float32)

    stationNumber = allTrainData.shape[2]

    stationIDList = getStationIDList()

    uncertaintyMeasure = []
    factors1 = []
    factors2 = []
    factors3 = []
    factors4 = []
    factors5 = []
    #
    # for rank in range(len(result2u[0])):
    #     targetStationIndex = rank
    #     demandMaskTensorFeed = np.array(demandMask[targetStationIndex]).reshape([stationNumber, 1])
    #     allTrainData = np.array(GraphValueMatrix[0: trainDataLength + valDataLength], dtype=np.float32)
    #     allTrainData = np.dot(allTrainData.reshape([-1, stationNumber]), demandMaskTensorFeed).reshape([-1, 24])
    #     allTestData = np.array(GraphValueMatrix[trainDataLength + valDataLength:], dtype=np.float32)
    #     allTestData = np.dot(allTestData.reshape([-1, stationNumber]), demandMaskTensorFeed).reshape([-1, 24])
    #
    #     demandList = allTrainData.reshape([-1, ])
    #
    #     allTrainDataVar = np.var(demandList)
    #
    #     allTrainDataSum = np.sum(allTrainData, axis=1)
    #
    #     if allTrainDataSum[rank] == 0:
    #         continue
    #
    #     print(rank, stationIDList[rank],
    #           getStationLocation(stationIDList[rank]),
    #           'demand', np.mean([e for e in allTrainDataSum if e != 0]),
    #           'RMSE', result2[1][rank],
    #           'CI', result2u[1][rank],
    #           result2u[1][rank] / np.mean(allTrainData.reshape([-1, ]))
    #           # 'RMSE/Max', result2[0][rank]/,
    #           # np.mean(allTrainDataVar),
    #           # result2[0][rank],
    #           # result2u[0][rank],
    #           # np.mean(allTrainDataVar) / result2u[0][rank],
    #           )
    #
    #     if rank == 2 or rank == 5:
    #
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot(1, 1, 1)
    #         X = np.arange(minLength) + 1
    #
    #         #[e/255 for e in colorList[::-1][int(step/(len(allTrainData)/len(colorList)))]]
    #         # /max(allTrainData.reshape([-1, ]))
    #
    #         # step = 0
    #         # for e in allTrainData[:-1]:
    #         #     if np.mean(e) == 0:
    #         #         continue
    #         #     ax1.plot(e, color=[0,0,1], lw=0.75)
    #         #     step += 1
    #         #
    #         # for e in allTestData[:-1]:
    #         #     if np.mean(e) == 0:
    #         #         continue
    #         #     ax1.plot(e, color=[1,0,0], lw=0.75)
    #         #     step += 1
    #
    #         minTrain = np.min(allTrainData, axis=0)
    #         maxTrain = np.max(allTrainData, axis=0)
    #
    #         avgTrain = np.average(allTrainData, axis=0)
    #
    #         ax1.plot([e for e in range(24)],
    #                  avgTrain,
    #                  color=[0, 0, 1], lw=2, label="Average train data")
    #
    #         ax1.fill_between([e for e in range(24)],
    #                          minTrain, maxTrain,
    #                          color='black', alpha=0.5, label="Train data interval")
    #
    #         # ax1.plot(allTrainData[-1], color=[0, 0, 1], lw=0.75, label="train data")
    #         # ax1.plot(allTestData[-1], color=[1, 0, 0], lw=0.75, label='test data')
    #
    #         font1 = {'family': 'Times New Roman',
    #                  'weight': 'normal',
    #                  'size': 60,
    #                  }
    #
    #         from matplotlib.pyplot import gca
    #
    #         a = gca()
    #         a.set_xticklabels(a.get_xticks(), font1)
    #         a.set_yticklabels(a.get_yticks(), font1)
    #
    #         # plt.xticks(fontsize=40)
    #         # plt.yticks(fontsize=40)
    #         ax1.legend(prop=font1)
    #         ax1.set_xlabel('Time(hour)', font1)
    #         ax1.set_ylabel('Check-in amount', font1)
    #         ax1.set_title('Check-in amount of training data', font1)
    #
    #         fig.set_size_inches(30, 15)
    #         fig.savefig(os.path.join(paperPath, 'checkIn-%s.jpg' % rank), dpi=50)
    #         plt.close()
    #
    #     uncertaintyMeasure.append(result2u[0][rank]) #/ np.mean([e for e in allTrainData if e != 0]))
    #
    #     factors1.append(np.mean([e for e in allTrainDataSum if e != 0]))
    #     factors2.append(allTrainDataVar)
    #     factors3.append(result2[0][rank])
    #     factors5.append(max(allTrainDataSum))
    #
    # np.savetxt(os.path.join(txtPath, 'factor1_demand.txt'), np.array(factors1), delimiter=' ', newline='\n')
    # np.savetxt(os.path.join(txtPath, 'factor2_demand.txt'), np.array(factors2), delimiter=' ', newline='\n')
    #
    # print(np.corrcoef(uncertaintyMeasure, factors1))
    #
    # print(np.corrcoef(uncertaintyMeasure, factors2))
    #
    # print(np.corrcoef(uncertaintyMeasure, factors3))
    #
    # print(np.corrcoef(uncertaintyMeasure, factors5))