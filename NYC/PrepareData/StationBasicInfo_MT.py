from localPath import rawBikeDataPath
import os
import csv
import json
from dateutil.parser import parse
from DataAPI.utils import saveJsonData, getJsonData
from APIS.multi_threads import multipleProcess

csvFileNameList = [e for e in os.listdir(rawBikeDataPath) if e.endswith(".csv")]

def compareTime(startTime, oldTime):
    new = parse(startTime)
    old = parse(oldTime)
    if new < old:
        return True
    else:
        return False

def checkLatLng(lat, lng):
    if lat == 0 or lng == 0 or lat > 45:
        return False
    else:
        return True

"""
partitionFunc = lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i]
"""

def task(ShareQueue, Locker, myCsvFileNameList, parameterList):
    stationAppearTime = {}
    for csvFile in myCsvFileNameList:
        with open(os.path.join(rawBikeDataPath, csvFile)) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            print(csvFile)
            for row in f_csv:
                # get all the data
                startTime = row[1]
                stopTime = row[2]
                startStationID = row[3]
                endStationID = row[7]

                startStationLat = float(row[5])
                startStationLong = float(row[6])
                
                endStationLat = float(row[9])
                endStationLong = float(row[10])

                # get the appearTime
                if checkLatLng(startStationLat, startStationLong):
                    if startStationID not in stationAppearTime:
                        startStationName = row[4]
                        stationAppearTime[startStationID] = [startTime, startStationLat, startStationLong, startStationName]
                    elif compareTime(startTime, stationAppearTime[startStationID][0]):
                        stationAppearTime[startStationID] = [startTime, startStationLat, startStationLong, startStationName]
                if checkLatLng(endStationLat, endStationLong):
                    if endStationID not in stationAppearTime:
                        endStationName = row[8]
                        stationAppearTime[endStationID] = [stopTime, endStationLat, endStationLong, endStationName]
                    elif compareTime(stopTime, stationAppearTime[endStationID][0]):
                        stationAppearTime[endStationID] = [stopTime, endStationLat, endStationLong, endStationName]

    Locker.acquire()
    ShareQueue.put(stationAppearTime)
    Locker.release()

def reduceFunction(a, b):
    for key, value in b.items():
        if key not in a:
            a[key] = value
        else:
            if compareTime(value[0], a[key][0]):
                a[key] = value
    return a


if __name__ == '__main__':

    n_jobs = 11

    partitionFunc = lambda csvFileNameList, i, n_job: [csvFileNameList[e] for e in range(len(csvFileNameList)) if e % n_job == i]

    stationAppearTimeDict = multipleProcess(csvFileNameList, partitionFunc, task, n_jobs, reduceFunction, [])

    saveJsonData(stationAppearTimeDict, "stationAppearTime.json")

    # stationAppearTimeDict = getJsonData("stationAppearTime.json")

    stationInformation = {}
    for stationID in stationAppearTimeDict.keys():
        stationInformation[stationID] = parse(stationAppearTimeDict[stationID][0])
    stationInformation = sorted(stationInformation.items(), key=lambda x:x[1], reverse=False)
    saveJsonData({'stationID': [e[0] for e in stationInformation],
                   'buildTime': [e[1].strftime('%Y-%m-%d %H:%M:%S') for e in stationInformation]},
                 'stationIdOrderByBuildTime.json')

    print(len(stationInformation))