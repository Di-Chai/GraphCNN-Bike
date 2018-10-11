import numpy as np
from localPath import *
import os
from SharedParameters.SharedParameters import *
from DataAPI.utils import checkZero, RMSE_OP

testDayLength = 14

def ShareLSTMResult(codeVersion):
    # load result
    finalPreResult = np.loadtxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-finalPreResult.txt'), delimiter=' ')
    uncertainty = np.loadtxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-uncertainty.txt'), delimiter=' ')
    testTarget = np.loadtxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-testTarget.txt'), delimiter=' ')

    stationNumber = finalPreResult.shape[1]

    RMSEList = []
    for i in range(stationNumber):
        # realPrediction = []
        # realTarget = []
        # zeroList = np.zeros([24 - featureLength - targetLength + 1])
        # for j in range(0, finalPreResult.shape[0], 24 - featureLength - targetLength + 1):
        #     startIndex = j
        #     endIndex = j + 24 - featureLength - targetLength + 1
        #
        #     if checkZero(testTarget[startIndex:endIndex, i]) is False:
        #         realPrediction.append(finalPreResult[startIndex:endIndex, i])
        #         realTarget.append(testTarget[startIndex:endIndex, i])
        #     else:
        #         realPrediction.append(zeroList)
        #         realTarget.append(zeroList)
        #
        #
        # RMSE = RMSE_OP(realPrediction, realTarget)
        RMSEList.append(RMSE_NoZero(testTarget[:testDayLength*(24-featureLength-targetLength+1), i],
                                    finalPreResult[:testDayLength*(24-featureLength-targetLength+1), i]))
    
    print('Top 5', np.mean(RMSEList[:5]))
    print('Top 10', np.mean(RMSEList[:10]))
    print('Top 30', np.mean(RMSEList[:30]))

    print(RMSEList)

def RMSE_NoZero(target, predict):
    realPrediction = []
    realTarget = []
    zeroList = np.zeros([24 - featureLength - targetLength + 1])
    for j in range(0, predict.shape[0], 24 - featureLength - targetLength + 1):
        startIndex = j
        endIndex = j + 24 - featureLength - targetLength + 1
        if checkZero(target[startIndex:endIndex]) is False:
            realPrediction.append(predict[startIndex:endIndex])
            realTarget.append(target[startIndex:endIndex])
        else:
            realPrediction.append(zeroList)
            realTarget.append(zeroList)

    RMSE = RMSE_OP(realPrediction, realTarget)
    return RMSE

if __name__ == '__main__':
    # ShareLSTMResult('AllStationBasedModel-SimplePreNN')
    # ShareLSTMResult('AllStationBasedModel-Partial-GCNLSTM')
    ShareLSTMResult('AllStationBasedModel-ShareLSTM')