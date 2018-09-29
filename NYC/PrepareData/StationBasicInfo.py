from localPath import rawBikeDataPath
import os
import csv
import json
from dateutil.parser import parse
from DataAPI.utils import saveJsonData, getJsonData

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

stationAppearTime = {}
for csvFile in csvFileNameList:
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

with open('stationAppearTime.json', 'w') as f:
    json.dump(stationAppearTime, f)