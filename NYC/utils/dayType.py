import datetime
from localPath import *
import json
import os
from dateutil.parser import parse
import types

publicHolidayList = ['01-01', '01-02', '01-16', '02-12', '02-13', '02-20', '05-29', '07-04', '09-04',
                     '10-09', '11-10', '11-11', '11-23', '12-25']

try:
    with open(os.path.join(jsonPath, 'weatherDict.json'), 'r') as f:
        weatherDict = json.load(f)
except:
    weatherDict = {}

def isBadDay(dateString):
    isBadDayDict = weatherDict['isBadDay']
    date = parse(dateString).strftime('%Y-%m-%d')
    if date not in isBadDayDict:
        return -1
    else:
        if isBadDayDict[date] == 1:
            return True
        else:
            return False

def isWorkDay(dateString):
    date = parse(dateString)
    if date.strftime('%m-%d') in publicHolidayList:
        return False
    week = date.weekday()
    if week < 5:
        return True
    else:
        return False

def inBuildPhase(dateString):
    date = parse(dateString)
    phase = [['2015-7-29', '2015-11-6'], ['2016-7-20', '2016-10-4'],
             ['2017-9-8', '2017-10-1']]
    for phaseRange in phase:
        if date > parse(phaseRange[0]) and date < parse(phaseRange[1]):
            return True
    return False

def isOverlapWithBuildPhase(dateStringList):
    result = False
    for e in dateStringList:
        if inBuildPhase(e):
            result = True
            break
    return result