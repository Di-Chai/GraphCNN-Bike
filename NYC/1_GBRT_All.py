from dataAPI.utils import *
import tensorflow as tf
from scipy import stats
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

GraphValueData = getJsonData('GraphValueMatrix.json')

GraphValueMatrix = np.array(GraphValueData['GraphValueMatrix'], dtype=np.float32)
tem = np.array(GraphValueData['tem'], dtype=np.float32)
wind = np.array(GraphValueData['wind'], dtype=np.float32)

stationIDDict = getJsonData('centralStationIDList.json')
centralStationIDList = stationIDDict['centralStationIDList']
allStationIDList = stationIDDict['allStationIDList']

# time feature
timeRange = ['2013-07-01', '2017-09-30']
date = parse(timeRange[0])
endData = parse(timeRange[1])
dateStringList = []
while date <= endData:
    dateString = date.strftime(dateTimeMode)
    if isWorkDay(dateString) and isBadDay(dateString) == False:
        dateStringList.append(dateString)
    date = date + datetime.timedelta(days=1)

testDataLength = 80
featureLength = 6
targetLength = 1

def moveSample(demandData, temData, windData, dateList):
    Feature0 = []
    Target0 = []
    for j in range(len(demandData)):
        dailyRecord = demandData[j]
        for k in range(len(dailyRecord) - featureLength - targetLength + 1):
            Feature0.append([e for e in dailyRecord[k: k + featureLength]] +
                                  [temData[len(temData) - len(demandData) + j][k],
                                   windData[len(windData) - len(demandData) + j][k],
                                  ]
                             + [k+featureLength, parse(dateList[j]).weekday()]
                            )
            Target0.append(float(dailyRecord[k + featureLength: k + featureLength + targetLength]))
    return Feature0, Target0

seriesData = np.sum(GraphValueMatrix, axis=2)

trainData = seriesData[:-testDataLength]
trainTem = tem[:-testDataLength]
trainWind = wind[:-testDataLength]
trainDateStringList = dateStringList[:-testDataLength]

testData = seriesData[-testDataLength:]
testTem = tem[-testDataLength:]
testWind = wind[-testDataLength:]
testDateStringList = dateStringList[-testDataLength:]

trainFeature, trainTarget = moveSample(trainData, trainTem, trainWind, trainDateStringList)

testFeature, testTarget = moveSample(testData, testTem, testWind, testDateStringList)

GBRT = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                max_depth = 10, random_state = 0,
                                loss = 'ls').fit(trainFeature, trainTarget)

testPredict = GBRT.predict(testFeature)

RMSE = np.mean((testPredict - np.array(testTarget, dtype=np.float32))**2)**0.5

print(RMSE)

######################################################################################################################
# proportion learning

# 1 parameters
H = 30 * 24
P1 = 0.9
P2 = 0.9
theta1 = 5
theta2 = 5

predictResult = []
counter = 0
predictLength = 80
for i in range(predictLength*24):
    if i % 24 < featureLength:
        continue
    proportionList = []
    weightList = []
    dayIndexStart = len(dateStringList) - testDataLength + int(i / 24)
    hourIndexStart = i % 24
    for j in range(H):
        if j % 24 < featureLength:
            continue
        dayIndex = len(dateStringList) - testDataLength + int(i/24) - int(j/24)
        hourIndex = j % 24
        if np.sum(GraphValueMatrix[dayIndex][hourIndex]) != 0:
            proportionList.append(GraphValueMatrix[dayIndex][hourIndex] / np.sum(GraphValueMatrix[dayIndex][hourIndex]))
        else:
            proportionList.append(GraphValueMatrix[dayIndex][hourIndex])

        r = lambda t1,t2: abs(t1 - t2) % 24
        deltaH = min(r(j, 0), 24 - r(j, 0))
        deltaD = int(j / 24)
        lambda1 = np.power(P1, deltaH) * np.power(P2, deltaD)

        lambda3 = (1 / (2 * np.pi * theta1 * theta2)) * np.exp(-((tem[dayIndexStart][hourIndexStart] -
                                                                  tem[dayIndex][hourIndex])**2/(theta1**2) +
                                                                 (wind[dayIndexStart][hourIndexStart] -
                                                                  wind[dayIndex][hourIndex])**2/(theta2**2)))
        weightList.append(lambda1 * lambda3)

    finalResult = np.sum([proportionList[e]*weightList[e] for e in range(len(proportionList))], axis=0) / np.sum(weightList)
    predictResult.append(finalResult * testPredict[counter])
    counter += 1

predictResult = np.array(predictResult, dtype=np.float32)

# RMSE
dayLength = 14
counterForRMSED = 0
RMSEList = []
for stationID in centralStationIDList:
    stationIndex = allStationIDList.index(stationID)
    a = predictResult[0:dayLength*(24-6), stationIndex].reshape([-1,])
    b = GraphValueMatrix[-80: -80 + dayLength, :24 - featureLength, stationIndex].reshape([-1,])

    if checkZero(b) == False:
        a_nonZero = []
        b_nonZero = []
        for i in range(dayLength):
            tmp_a = [e for e in a[i*(24-6): (i+1)*(24-6)]]
            tmp_b = [e for e in b[i*(24-6): (i+1)*(24-6)]]
            if checkZero(tmp_b) == False:
                a_nonZero = a_nonZero + tmp_a
                b_nonZero = b_nonZero + tmp_b

        a_nonZero = np.array(a_nonZero, dtype=np.float32)
        b_nonZero = np.array(b_nonZero, dtype=np.float32)

        RMSEList.append(np.mean((a_nonZero - b_nonZero)**2)**0.5)
    counterForRMSED += 1
    if counterForRMSED == 30:
        break
print(RMSEList)
print(np.mean(RMSEList[0:5]),
      np.mean(RMSEList[0:10]),
      np.mean(RMSEList[0:30]))













