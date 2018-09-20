# -*-  coding:utf-8 -*-
from localPath import *
from utils.symbols import *
from utils.distance import haversine
from utils.dayType import *
from dateutil.parser import parse
import numpy as np
import csv
from localPath import *
import json
import random
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
from multiprocessing import Pool
from functools import reduce

dateTimeMode = '%Y-%m-%d'

def checkZero(valueList):
    result = True
    for e in valueList:
        if e != 0:
            result = False
            break
    return result

def saveJsonData(dataDict, fileName):
    with open(os.path.join(jsonPath, fileName), 'w') as f:
        json.dump(dataDict, f)
    print('Saved', fileName)

def getJsonData(fileName):
    with open(os.path.join(jsonPath, fileName), 'r') as f:
        data = json.load(f)
    print('load', fileName)
    return data

def getJsonDataFromPath(fullPath, showMessage=True):
    with open(fullPath, 'r') as f:
        data = json.load(f)
    if showMessage:
        print('load', fullPath)
    return data

def saveJsonDataToPath(dataDict, fullPath):
    with open(fullPath, 'w') as f:
        json.dump(dataDict, f)
    print('Saved', fullPath)

if os.path.isfile(os.path.join(jsonPath, 'stationAppearTime.json')) is False:
    stationInfoFile = [e for e in os.listdir(ChicagoCSVDataPath) if 'Stations' in e and e.endswith('.csv')]
    stationAppearTimeDict = {}
    for file in stationInfoFile:
        with open(os.path.join(ChicagoCSVDataPath, file), 'r') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            IDIndex  = None
            NameIndex = None
            LatIndex = None
            LngIndex = None
            BuildTimeIndex = None
            for e in headers:
                if 'id' in e:
                    IDIndex = headers.index(e)
                if 'name' in e:
                    NameIndex = headers.index(e)
                if 'lat' in e:
                    LatIndex = headers.index(e)
                if 'long' in e:
                    LngIndex = headers.index(e)
                if 'date' in e:
                    BuildTimeIndex = headers.index(e)
            print(file)
            step = 0
            for row in f_csv:
                stationID = row[IDIndex]
                if stationID not in stationAppearTimeDict:
                    stationAppearTimeDict[stationID] = [None, None, None, None]

                stationName = None if NameIndex is None else row[NameIndex]
                lat = None if LatIndex is None else row[LatIndex]
                lng = None if LngIndex is None else row[LngIndex]
                buildTime = None if BuildTimeIndex is None else row[BuildTimeIndex]

                currentlist = [buildTime, lat, lng, stationName]

                for e in range(len(currentlist)):
                    if currentlist[e] is not None or stationAppearTimeDict[stationID] is None:
                        stationAppearTimeDict[stationID][e] = currentlist[e]

    stationIDListOrderByBuildTime = [e[0] for e in sorted(stationAppearTimeDict.items(), key=lambda x:parse(x[1][0]), reverse=False)]

    saveJsonData(stationAppearTimeDict, 'stationAppearTime.json')
    saveJsonData({'stationIDListOrderByBuildTime': stationIDListOrderByBuildTime}, 'stationIDListOrderByBuildTime.json')

stationAppearTimeDict = getJsonData('stationAppearTime.json')
stationIDListOrderByBuildTime = getJsonData('stationIDListOrderByBuildTime.json')['stationIDListOrderByBuildTime']

def getStationIDList():
    return stationIDListOrderByBuildTime

def getStationLocation(stationID):
    currentStationInfo = stationAppearTimeDict[stationID]
    lat = float(currentStationInfo[1])
    lng = float(currentStationInfo[2])
    return lat, lng

def computeDistanceBetweenAB(stationID_A, stationID_B):
    stationID_A_Info = stationAppearTimeDict[stationID_A]
    stationID_B_Info = stationAppearTimeDict[stationID_B]
    return haversine(float(stationID_A_Info[2]), float(stationID_A_Info[1]),
                     float(stationID_B_Info[2]), float(stationID_B_Info[1]))

def insertEmptyData(valueList, errorValue):
    for i in range(valueList.__len__()):
        if valueList[i] == errorValue:
            neighbourValue = []
            j = i - 1
            while j >= 0 and valueList[j] == errorValue:
                j -= 1
            if j >= 0:
                neighbourValue.append(valueList[j])
            j = i + 1
            while j < valueList.__len__() and valueList[j] == errorValue:
                j += 1
            if j < valueList.__len__():
                neighbourValue.append(valueList[j])
            if neighbourValue.__len__() != 0:
                valueList[i] = np.mean(neighbourValue)
    return valueList

def removeJsonData(fileName):
    os.remove(os.path.join(jsonPath, fileName))

if __name__ == '__main__':
    getStationIDList()