from DataAPI.utils import *
from DataAPI.apis import d_diff
from SharedParameters.SharedParameters import *

import csv

stationIDDict = getJsonData('centralStationIDList.json')
centralStationIDList = stationIDDict['centralStationIDList']
allStationIDList = stationIDDict['allStationIDList']

stationAppearTimeDict = getJsonData('stationAppearTime.json')

GraphValueData = getJsonData('GraphValueMatrix.json')

GraphValueMatrix = GraphValueData['GraphValueMatrix']
tem = GraphValueData['tem']
wind = GraphValueData['wind']

# file 1 location information
# [[lat, lng], [], ...]

locationList = []
for stationID in centralStationIDList:
    stationInfo = stationAppearTimeDict.get(stationID, None)
    if stationInfo is not None:
        locationList.append([stationInfo[2], stationInfo[1]])

with open(os.path.join(txtPath, 'locationInfo.txt'), 'w') as f:
    for element in locationList:
        f.write(' '.join(element) + '\n')

with open(os.path.join(txtPath, 'locationInfo.csv'), "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(locationList)

# file 2 bike flow information
flowList = []

for dayValue in GraphValueMatrix[:-testDataLength]:
    for value in dayValue:
        newHourValue = []
        for i in range(len(centralStationIDList)):
            newHourValue.append(value[allStationIDList.index(centralStationIDList[i])])
        flowList.append(newHourValue)

with open(os.path.join(txtPath, 'flowList.txt'), 'w') as f:
    for element in flowList:
        f.write(' '.join([str(e) for e in element]) + '\n')
with open(os.path.join(txtPath, 'flowList.csv'), "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(flowList)