from DataAPI.utils import *
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

testTarget = np.array(GraphValueMatrix[-testDataLength:], dtype=np.float32)

predictionResult = []
predictDayLength = 14
for i in range(predictDayLength):
    pointer = len(dateStringList) - testDataLength + i

    currentDateString = dateStringList[pointer]
    candidateList = []
    for j in range(pointer):
        if parse(currentDateString).weekday() == parse(dateStringList[j]).weekday():
            candidateList.append(GraphValueMatrix[j])

    candidateList = np.array(candidateList)
    predictionResult.append(np.mean(candidateList, axis=0)[6:, :])

predictionResult = np.array(predictionResult, dtype=np.float32)
corrTestTarget = testTarget[:predictDayLength, 6:, :]

RMSEList = []
targetStationNumbers = 30
for i in range(targetStationNumbers):
    stationIndex = allStationIDList.index(centralStationIDList[i])

    if checkZero(corrTestTarget[:, :, stationIndex].reshape([-1, ])) == False:

        a = predictionResult[:, :, stationIndex].reshape([-1, ])
        b = corrTestTarget[:, :, stationIndex].reshape([-1, ])

        a_nonZero = []
        b_nonZero = []
        for i in range(predictDayLength):
            tmp_a = [e for e in a[i*(24-6): (i+1)*(24-6)]]
            tmp_b = [e for e in b[i*(24-6): (i+1)*(24-6)]]
            if checkZero(tmp_b) == False:
                a_nonZero = a_nonZero + tmp_a
                b_nonZero = b_nonZero + tmp_b
        
        a_nonZero = np.array(a_nonZero, dtype=np.float32)
        b_nonZero = np.array(b_nonZero, dtype=np.float32)

        RMSE = np.mean((a_nonZero - b_nonZero)**2)**0.5
        RMSEList.append(RMSE)

print('Top 5', np.mean(RMSEList[:5]))
print('Top 10', np.mean(RMSEList[:10]))
print('Top 30', np.mean(RMSEList[:30]))