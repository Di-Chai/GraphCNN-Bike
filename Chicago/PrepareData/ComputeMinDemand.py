# -*-  coding:utf-8 -*-
from localPath import *
import os
import csv
import json
from dateutil.parser import parse
from DataAPI.utils import getJsonData, saveJsonDataToPath
from APIS.multi_threads import multipleProcess
from DataAPI.apis import getRawBikeDataFileList
dateTimeMode = '%Y-%m-%d'


stationIdOrderByBuildTime = getJsonData('stationIdOrderByBuildTime.json')
stationIdList = stationIdOrderByBuildTime['stationID']

# multiple threads
# 1 distribute list
distributeList = stationIdList
# 2 partition function
partitionFunc = lambda dtList, i, n_job: [dtList[e] for e in range(len(dtList)) if e % n_job == i]
# 3 n_jobs
n_jobs = 8
# 4 reduce function
def reduceFunction(a, b):
    for key, value in b.items():
        a[key] = value
    return a
# 5 task function
def task(ShareQueue, Locker, distributedList, parameterList):
    csvFileNameList = getRawBikeDataFileList()
    finalResult = {}
    for csvFile in csvFileNameList:
        with open(os.path.join(rawBikeDataPath, csvFile), 'r') as f:
            print(csvFile)
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                startStationID = row[3]
                endStationID = row[7]
                if startStationID in distributedList:
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
                if endStationID in distributedList:
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
    print('Process Finish')
    # Locker.acquire()
    ShareQueue.put(finalResult)
    # Locker.release()
# 6 parameter list
parameterList = []
# 7 run !
if __name__ == '__main__':
    demandResult = multipleProcess(distributeList=distributeList, partitionDataFunc=partitionFunc, taskFunction=task,
                            n_jobs=n_jobs, reduceFunction=reduceFunction, parameterList=parameterList)
    # save data
    for keys in demandResult:
        saveJsonDataToPath(demandResult[keys], os.path.join(demandMinDataPath, '%s.json' % keys))