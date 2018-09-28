from dataAPI.utils import *
from sharedParametersV2 import *
import sys
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

def adfTest(timeSeries, maxlags=None, printFlag=False):
    t = sm.tsa.stattools.adfuller(timeSeries, maxlag=maxlags)
    output = pd.DataFrame(
        index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
               "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    if printFlag:
        print(output)
    return t

def d_diff(value, valueList):
    resultList = []
    for i in range(valueList.__len__()):
        resultList.append(value + sum(valueList[:i + 1]))
    return resultList


argvList = sys.argv
currentFileName = argvList[1]
# currentFileName = 'GraphSingleStationDemandPreV7_0'
codeVersion = currentFileName.replace('_', '-')
rank = int(currentFileName.split('_')[-1])

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
targetStationIndex = rank
demandMaskTensorFeed = np.array(demandMask[targetStationIndex]).reshape([stationNumber, 1])

allTrainData = np.dot(allTrainData.reshape([-1, stationNumber]), demandMaskTensorFeed).reshape([-1,])
allTestData = np.dot(allTestData.reshape([-1, stationNumber]), demandMaskTensorFeed).reshape([-1,])

########################################################################################################################
trainDateRange = ['2013-07-01 00:00:00', '2014-12-07 00:00:00']
date = parse(trainDateRange[0])
trainIndexList = []
while date < parse(trainDateRange[1]):
    trainIndexList.append(date.strftime('%Y-%m-%d %H:%M:%S'))
    date = date + datetime.timedelta(seconds=3600)

testDateRange = ['2014-12-07 00:00:00', '2015-2-25 00:00:00']
date = parse(testDateRange[0])
testIndexList = []
while date < parse(testDateRange[1]):
    testIndexList.append(date.strftime('%Y-%m-%d %H:%M:%S'))
    date = date + datetime.timedelta(seconds=3600)

predictLength = 24

finalPreResult = []

for i in range(0, len(testIndexList[:24*14]), predictLength):
    print(codeVersion, 'Day:', int(i/predictLength))
    trainList = pd.Series(np.hstack((allTrainData, allTestData[0: i])))
    trainList.index = pd.Index(trainIndexList + testIndexList[0: i])

    predictDataRange = [testIndexList[i], testIndexList[i+predictLength-1]]

    # diff
    trainTimeSeries = trainList.diff(24)[24:]
    # train the model
    res = sm.tsa.stattools.arma_order_select_ic(trainTimeSeries, ic=['aic'])
    model = sm.tsa.ARMA(trainTimeSeries, res.aic_min_order).fit(disp=0)
    summary = (model.summary2(alpha=.05, float_format="%.8f"))
    print(summary)

    predict_data = model.predict(start=predictDataRange[0], end=predictDataRange[1], dynamic=False)
    predict_data_list = list(predict_data.values)

    realPredict = [trainList[-predictLength + e] + predict_data_list[e] for e in range(len(predict_data_list))]

    print("RMSE:", np.sqrt(np.mean((np.array(realPredict) - np.array(allTestData[i: i+predictLength]))**2)))

    finalPreResult = finalPreResult + realPredict

np.savetxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-finalPreResult.txt'),
               np.array(finalPreResult, dtype=np.float32),
               newline='\n', delimiter=' ')
np.savetxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-testTarget.txt'),
           allTestData, newline='\n', delimiter=' ')