# -*-  coding:utf-8 -*-
from localPath import *
import os
import csv
import json
from dateutil.parser import parse
dateTimeMode = '%Y-%m-%d'
import numpy as np

with open(os.path.join(jsonPath, 'stationIdOrderByBuildTime.json'), 'r') as f:
    stationIdOrderByBuildTime = json.load(f)
stationId = stationIdOrderByBuildTime['stationID']
stationBuildTime = stationIdOrderByBuildTime['buildTime']

local_id_list = []
for i in range(stationId.__len__()):
    if i % 8 == 4:
        local_id_list.append(stationId[i])

matrixSize = stationId.__len__()

stationIDMatrixIndex = {}
for i in range(stationId.__len__()):
    stationIDMatrixIndex[stationId[i]] = i

csvFileNameList = [e for e in os.listdir(csvDataPath) if e.endswith(".csv")]
# This will take a long time ...
finalResult = {}
for csvFile in csvFileNameList:
    with open(os.path.join(csvDataPath, csvFile)) as f:
        print(csvFile)
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            startStationID = row[3]
            endStationID = row[7]
            if startStationID in local_id_list:
                if startStationID not in finalResult:
                    finalResult[startStationID] = {}
                startTime = row[1]
                startTimeDate = parse(startTime).strftime(dateTimeMode)
                startTimeClock = parse(startTime).strftime('%H')
                if startTimeDate not in finalResult[startStationID]:
                    finalResult[startStationID][startTimeDate] = {}
                if startTimeClock not in finalResult[startStationID][startTimeDate]:
                    finalResult[startStationID][startTimeDate][startTimeClock] = {'in':{}, 'out':{}}
                if endStationID not in finalResult[startStationID][startTimeDate][startTimeClock]['out']:
                    finalResult[startStationID][startTimeDate][startTimeClock]['out'][endStationID] = 0
                finalResult[startStationID][startTimeDate][startTimeClock]['out'][endStationID] += 1
            if endStationID in local_id_list:
                if endStationID not in finalResult:
                    finalResult[endStationID] = {}
                stopTime = row[2]
                stopTimeDate = parse(stopTime).strftime(dateTimeMode)
                stopTimeClock = parse(stopTime).strftime('%H')
                if stopTimeDate not in finalResult[endStationID]:
                    finalResult[endStationID][stopTimeDate] = {}
                if stopTimeClock not in finalResult[endStationID][stopTimeDate]:
                    finalResult[endStationID][stopTimeDate][stopTimeClock] = {'in':{}, 'out':{}}
                if startStationID not in finalResult[endStationID][stopTimeDate][stopTimeClock]['in']:
                    finalResult[endStationID][stopTimeDate][stopTimeClock]['in'][startStationID] = 0
                finalResult[endStationID][stopTimeDate][stopTimeClock]['in'][startStationID] += 1

# save data
for keys in finalResult:
    with open(os.path.join(demandDataPath, '%s.json' % keys), 'w') as f:
        json.dump(finalResult[keys], f)


