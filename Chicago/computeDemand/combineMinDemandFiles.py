from dataAPI.apis import getJsonDataFromPath, saveJsonData, getStationIDList
import os
from localPath import demandMinDataPath

if __name__ == '__main__':
    minDemandDict = {
        stationID:getJsonDataFromPath(os.path.join(demandMinDataPath, stationID+'.json'))
        for stationID in getStationIDList()
    }
    saveJsonData(minDemandDict, 'minDemandDict.json')