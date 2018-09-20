import os
import json
import numpy as np
from utils.symbols import *
from localPath import *
from dateutil.parser import parse
from functools import reduce
from utils.distance import haversine
from multiprocessing import Pool
import csv
import datetime
from utils.dayType import *
import matplotlib.pyplot as plt

dateTimeMode = '%Y-%m-%d'

def getJsonData(fileName):
    with open(os.path.join(jsonPath, fileName), 'r') as f:
        data = json.load(f)
    print('load', fileName)
    return data

def saveJsonData(dataDict, fileName):
    with open(os.path.join(jsonPath, fileName), 'w') as f:
        json.dump(dataDict, f)
    print('Saved', fileName)

def removeJsonData(fileName):
    os.remove(os.path.join(jsonPath, fileName))

def checkZero(valueList):
    result = True
    for e in valueList:
        if e != 0:
            result = False
            break
    return result

stationAppearTimeDict = getJsonData('stationAppearTime.json')
stationIdOrderByBuiltTime = getJsonData('stationIdOrderByBuildTime.json')

def getBuildTime(stationID):
    return stationAppearTimeDict[stationID][0]

def getStationLocation(stationID):
    try:
        return [float(stationAppearTimeDict[stationID][1]), float(stationAppearTimeDict[stationID][2])]
    except:
        print('Can not find %s in stationAppearTimeDict' % stationID)

def computeDistanceBetweenAB(stationID_A, stationID_B):
    stationID_A_Info = stationAppearTimeDict[stationID_A]
    stationID_B_Info = stationAppearTimeDict[stationID_B]
    return haversine(float(stationID_A_Info[2]), float(stationID_A_Info[1]),
                     float(stationID_B_Info[2]), float(stationID_B_Info[1]))

def getStationIDList():
    return stationIdOrderByBuiltTime['stationID']

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

def check_negative(valueList):
    negativeFlag = False
    for value in valueList:
        if value < 0:
            negativeFlag = True
            break
    return negativeFlag


def check_tem_normal(dataList):
    normalFlag = False
    for e in dataList:
        if e != NO_TEM:
            normalFlag = True
            break
    return normalFlag


def get_positive_mean(dataList):
    sumList = []
    for e in dataList:
        if e > 0:
            sumList.append(e)
    if sumList.__len__() == 0:
        return EMPTY_DATA
    else:
        return np.mean(sumList)

def deletePNGFromPath(path):
    pngFileNameList = [e for e in os.listdir(path) if e.endswith(".png")]
    for e in pngFileNameList:
        os.remove(os.path.join(path, e))

def clearFolder(folderPath):
    fileAboutToRemove = os.listdir(folderPath)
    for file in fileAboutToRemove:
        os.chmod(os.path.join(folderPath, file), 777)
        os.remove(os.path.join(folderPath, file))


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


def flatList(valueList):
    resultList = valueList[0]
    for i in range(1, valueList.__len__()):
        resultList = resultList + valueList[i]
    return resultList


def combineDict(dict1, dict2):
    resultDict = {}
    for key in dict1:
        resultDict[key] = dict1[key]
    for key in dict2:
        resultDict[key] = dict2[key]
    return resultDict


def checkAllPositive(valueList):
    result = True
    for e in valueList:
        if e < 0:
            result = False
            break
    return result


def getBinaryLabels(value):
    if value == 0:
        return [1, 0]
    else:
        return [0, 1]

def combineList(valueList1, valueList2):
    resultList = valueList1
    if type(valueList2) == list:
        for e in valueList2:
            resultList.append(e)
    else:
        resultList.append(valueList2)
    return resultList


def combineMultipleElementList(valueList):
    head = valueList[0]
    for i in range(1, valueList.__len__()):
        head = combineList(head, valueList[i])
    return head

def flattenList(listValue):
    if listValue.__len__() <= 1:
        return listValue
    else:
        resultList = listValue[0]
        for i in range(1, listValue.__len__()):
            resultList = combineList(resultList, listValue[i])
        return resultList


def findFirstPositiveElement(valueList):
    pointer = 0
    while valueList[pointer] < 0 and pointer < valueList.__len__():
        pointer += 1
    return pointer

def deleteNegativePart(x, y):
    for i in range(y.__len__()):
        pointer = y.__len__() - 1 - i
        if y[pointer] < 0:
            del y[pointer]
            del x[pointer]
    return x, y


def subtractList(a, b):
    resultList = []
    for element in a:
        if element not in b:
            resultList.append(element)
    return resultList

def getTransitionMatrixSlave(timeStringRange, stationIDList, p, myRank):
    start = parse(timeStringRange[0])
    end = parse(timeStringRange[1])
    inTimeRange = lambda x: True if x > start and x < end else False
    csvFileNameList = [e for e in os.listdir(csvDataPath) if e.endswith(".csv")]
    transitionMatrix = [[0 for _ in range(stationIDList.__len__())] for _ in range(stationIDList.__len__())]
    for csvFile in csvFileNameList:
        with open(os.path.join(csvDataPath, csvFile)) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            print(csvFile)
            step = 0
            for row in f_csv:
                if step % p == myRank:
                    # get all the data
                    startTime = parse(row[1])
                    stopTime = parse(row[2])
                    startStationID = row[3]
                    endStationID = row[7]
                    if inTimeRange(startTime) and inTimeRange(stopTime) and startStationID in stationIDList and\
                        endStationID in stationIDList:
                        transitionMatrix[stationIDList.index(startStationID)][stationIDList.index(endStationID)] += 1
                step += 1
    np.savetxt(os.path.join(txtPath, 'transitionMatrix-%s.txt' % myRank), np.array(transitionMatrix, dtype=np.int32), delimiter=' ', newline='\n')

def getTransitionMatrixMulThreads(timeStringRange, stationIDList, n_jobs):
    p = Pool()
    for i in range(n_jobs):
        p.apply_async(getTransitionMatrixSlave, args=(timeStringRange, stationIDList, n_jobs, i))
    p.close()
    p.join()
    transitionMatrixList = []
    for i in range(n_jobs):
        transitionMatrixList.append(np.loadtxt(os.path.join(txtPath, 'transitionMatrix-%s.txt' % i), delimiter=' '))
    transitionMatrix = reduce(lambda x,y: x+y, transitionMatrixList)
    np.savetxt(os.path.join(txtPath, 'transitionMatrix.txt'), np.array(transitionMatrix, dtype=np.int32),
               delimiter=' ', newline='\n')


def getAdjustList(stationIDList, stationLabelList, distanceThreshold=1000):
    stationLabelDict = dict([[stationIDList[e], stationLabelList[e]] for e in range(stationIDList.__len__())])
    # get the transition matrix
    transitionMatrix = np.loadtxt(os.path.join(txtPath, 'transitionMatrix.txt'), delimiter=' ')
    # 1 get the neighbour clusters -> the nearest distance smaller than 1000m
    adjustList = []
    for i in range(stationIDList.__len__()):
        stationID = stationIDList[i]
        minDistance = []
        for j in range(stationIDList.__len__()):
            otherStationID = stationIDList[j]
            if stationLabelDict[stationID] == stationLabelDict[otherStationID]:
                continue
            if len(minDistance) == 0:
                minDistance = [stationLabelDict[otherStationID], computeDistanceBetweenAB(stationID, otherStationID)]
            else:
                distance = computeDistanceBetweenAB(stationID, otherStationID)
                if distance < minDistance[1]:
                    minDistance = [stationLabelDict[otherStationID], distance]
        if minDistance[1] < distanceThreshold:
            adjustList.append([stationID, (stationLabelDict[stationID], minDistance[0]), minDistance[1]])
    # 2 check the transition matrix to verify the adjust
    initLength = adjustList.__len__()
    for i in range(adjustList.__len__()):
        pointer = initLength - 1 - i
        adjust = adjustList[pointer]
        oldCluster = adjust[1][0]
        newCluster = adjust[1][1]
        stationIDIndex = stationIDList.index(adjust[0])
        oldClusterStationIDIndexList = [e for e in range(len(stationIDList)) if
                                   stationLabelList[e] == oldCluster and e != stationIDIndex]
        newClusterStationIDIndexList = [e for e in range(len(stationIDList)) if
                                   stationLabelList[e] == newCluster]
        transitionToOldCluster = (np.sum(transitionMatrix[stationIDIndex, oldClusterStationIDIndexList]) +
                                 np.sum(transitionMatrix[oldClusterStationIDIndexList, stationIDIndex])) / (
            [e for e in transitionMatrix[stationIDIndex, oldClusterStationIDIndexList] if e > 0].__len__() +
            [e for e in transitionMatrix[oldClusterStationIDIndexList, stationIDIndex] if e > 0].__len__()
        )
        transitionToNewCluster = (np.sum(transitionMatrix[stationIDIndex, newClusterStationIDIndexList]) +
                                 np.sum(transitionMatrix[newClusterStationIDIndexList, stationIDIndex])) / (
            [e for e in transitionMatrix[stationIDIndex, newClusterStationIDIndexList] if e > 0].__len__() +
            [e for e in transitionMatrix[newClusterStationIDIndexList, stationIDIndex] if e > 0].__len__()
        )
        if transitionToNewCluster < transitionToOldCluster:
            del adjustList[pointer]
    return adjustList

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2-0.3, height, '%s' % float(height))
