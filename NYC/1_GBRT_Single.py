from dataAPI.utils import *
import tensorflow as tf
from scipy import stats
import sys
from sklearn.ensemble import GradientBoostingRegressor

GraphValueData = getJsonData('GraphValueMatrix.json')

GraphValueMatrix = np.array(GraphValueData['GraphValueMatrix'], dtype=np.float32)
tem = np.array(GraphValueData['tem'], dtype=np.float32)
wind = np.array(GraphValueData['wind'], dtype=np.float32)

stationIDDict = getJsonData('centralStationIDList.json')
centralStationIDList = stationIDDict['centralStationIDList']
allStationIDList = stationIDDict['allStationIDList']

testDataLength = 80
featureLength = 6
targetLength = 1

def moveSample(demandData, temData, windData):
    Feature0 = []
    Target0 = []
    for j in range(len(demandData)):
        dailyRecord = demandData[j]
        for k in range(len(dailyRecord) - featureLength - targetLength + 1):
            Feature0.append([e for e in dailyRecord[k: k + featureLength]] +
                                  [temData[len(temData) - len(demandData) + j][k],
                                   windData[len(windData) - len(demandData) + j][k],
                                  ])
            Target0.append(float(dailyRecord[k + featureLength: k + featureLength + targetLength]))
    return Feature0, Target0

RMSECalTime = 14  # days
RMSEList = []
counter = 0
for stationCounter in range(GraphValueMatrix.shape[2]):

    if counter > 30:
        break
    counter += 1
    print(counter)

    seriesData = GraphValueMatrix[:, :, allStationIDList.index(centralStationIDList[stationCounter])].reshape([604, 24])

    trainData = seriesData[:-80]
    trainTem = tem[:-80]
    trainWind = wind[:-80]

    testData = seriesData[-80:]
    testTem = tem[-80:]
    testWind = wind[-80:]

    trainFeature, trainTarget = moveSample(trainData, trainTem, trainWind)

    testFeature, testTarget = moveSample(testData, testTem, testWind)

    GBRT = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1,
                                    max_depth = 10, random_state = 0,
                                    loss = 'ls').fit(trainFeature, trainTarget)
    testPredict = GBRT.predict(testFeature)

    a = (testPredict - np.array(testTarget, dtype=np.float32))**2

    if checkZero(testTarget) == False:
        RMSE = []
        for i in range(RMSECalTime):
            if checkZero(testTarget[i*(24-6):(i+1)*(24-6)]) == False:
                RMSE = RMSE + [e for e in (testPredict[i*(24-6):(i+1)*(24-6)] - np.array(testTarget[i*(24-6):(i+1)*(24-6)], dtype=np.float32))**2]
        RMSE = np.mean(RMSE)**0.5
        RMSEList.append(RMSE)

print('Top 5', np.mean(RMSEList[:5]))
print('Top 10', np.mean(RMSEList[:10]))
print('Top 30', np.mean(RMSEList[:30]))










