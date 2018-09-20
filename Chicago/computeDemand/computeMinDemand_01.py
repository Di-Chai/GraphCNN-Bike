# -*-  coding:utf-8 -*-
from localPath import *
import os
import csv
import json
from dateutil.parser import parse
from dataAPI.apis import *
dateTimeMode = '%Y-%m-%d'

stationIDList = getStationIDList()

local_id_list = []
currentPath = os.path.dirname(os.path.abspath(__file__))
allFileList = [e for e in os.listdir(currentPath) if e.startswith('computeMinDemand_') and e.endswith('.py')]
totalRank = allFileList.__len__()
fileName = __file__
fileName = fileName.split('/')[-1][:-3]
my_rank = int(fileName.split('_')[-1])
for i in range(stationIDList.__len__()):
    if i % totalRank == my_rank:
        local_id_list.append(stationIDList[i])

csvFileNameList = [e for e in os.listdir(ChicagoCSVDataPath) if e.endswith(".csv") and 'Trips' in e]
finalResult = {}
for csvFile in csvFileNameList:
    with open(os.path.join(ChicagoCSVDataPath, csvFile)) as f:
        print(csvFile)
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            startStationID = row[5]
            endStationID = row[7]
            if startStationID in local_id_list:
                if startStationID not in finalResult:
                    finalResult[startStationID] = {}
                startTime = row[1]
                startTimeData = parse(startTime)
                startTimeDateString = startTimeData.strftime(dateTimeMode)
                # 左闭右开
                startTimeMinute = startTimeData.hour * 60 + startTimeData.minute
                if startTimeDateString not in finalResult[startStationID]:
                    finalResult[startStationID][startTimeDateString] = {'in': {},
                                                                        'out': {},
                                                                        'inStation': {},
                                                                        'outStation': {}}
                if startTimeMinute not in finalResult[startStationID][startTimeDateString]['out']:
                    finalResult[startStationID][startTimeDateString]['out'][startTimeMinute] = 0
                if startTimeMinute not in finalResult[startStationID][startTimeDateString]['outStation']:
                    finalResult[startStationID][startTimeDateString]['outStation'][startTimeMinute] = []
                finalResult[startStationID][startTimeDateString]['out'][startTimeMinute] += 1
                finalResult[startStationID][startTimeDateString]['outStation'][startTimeMinute].append(endStationID)
            if endStationID in local_id_list:
                if endStationID not in finalResult:
                    finalResult[endStationID] = {}
                stopTime = row[2]
                stopTimeDate = parse(stopTime)
                stopTimeDateString = stopTimeDate.strftime(dateTimeMode)
                # 左闭右开
                stopTimeMinute = stopTimeDate.hour * 60 + stopTimeDate.minute
                if stopTimeDateString not in finalResult[endStationID]:
                    finalResult[endStationID][stopTimeDateString] = {'in': {},
                                                                        'out': {},
                                                                        'inStation': {},
                                                                        'outStation': {}}
                if stopTimeMinute not in finalResult[endStationID][stopTimeDateString]['in']:
                    finalResult[endStationID][stopTimeDateString]['in'][stopTimeMinute] = 0
                if stopTimeMinute not in finalResult[endStationID][stopTimeDateString]['inStation']:
                    finalResult[endStationID][stopTimeDateString]['inStation'][stopTimeMinute] = []
                finalResult[endStationID][stopTimeDateString]['in'][stopTimeMinute] += 1
                finalResult[endStationID][stopTimeDateString]['inStation'][stopTimeMinute].append(startStationID)
# save data
for keys in finalResult:
    with open(os.path.join(demandMinDataPath, '%s.json' % keys), 'w') as f:
        json.dump(finalResult[keys], f)