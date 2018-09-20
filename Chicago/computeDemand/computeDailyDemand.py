from loadData import get_stationIDList
from localPath import demandDataPath
import os
import json
from dataAPI.apis import *
from utils.symbols import *


def computeDailyDemand(stationID):
    dailyDemand = {}
    with open(os.path.join(demandDataPath, stationID + '.json'), 'r') as f:
        currentStationDemand = json.load(f)
    dateList = list(currentStationDemand.keys())
    for date in dateList:
        dateDemand = currentStationDemand[date]
        hourDemandTotal = 0
        for hour in dateDemand:
            for type in dateDemand[hour]:
                for targetStation in dateDemand[hour][type]:
                    hourDemandTotal += dateDemand[hour][type][targetStation]
        dailyDemand[date] = hourDemandTotal
    return dailyDemand


if __name__ == '__main__':
    stationIDList = get_stationIDList()
    dailyDemandDict = {}
    for stationID in stationIDList:
        print(stationID)
        dailyDemandDict[stationID] = computeDailyDemand(stationID)
    saveJsonData(dailyDemandDict, 'dailyDemandDict.json')