from localPath import *
import os
from APIS.csv_api import loadCSVFileFromPath
from DataAPI.utils import saveJsonData
from dateutil.parser import parse

if __name__ == '__main__':

    # n_jobs = 8
    #
    # partitionFunc = lambda csvFileNameList, i, n_job: [csvFileNameList[e] for e in range(len(csvFileNameList)) if e % n_job == i]
    #
    # stationAppearTimeDict = multipleProcess(csvFileNameList, partitionFunc, task, n_jobs, reduceFunction, [])
    #
    # saveJsonData(stationAppearTimeDict, "stationAppearTime.json")
    #
    # # stationAppearTimeDict = getJsonData("stationAppearTime.json")
    #

    header, file = loadCSVFileFromPath(os.path.join(rawBikeDataPath, 'Divvy_Stations_2017_Q3Q4.csv'), fileWithHeader=True)

    stationAppearTimeDict = {}

    for row in file:
        stationAppearTimeDict[row[0]] = [row[6], row[3], row[4], [row[1]]]

    saveJsonData(stationAppearTimeDict, "stationAppearTime.json")

    stationInformation = {}
    for stationID in stationAppearTimeDict.keys():
        stationInformation[stationID] = parse(stationAppearTimeDict[stationID][0])
    stationInformation = sorted(stationInformation.items(), key=lambda x:x[1], reverse=False)
    saveJsonData({'stationID': [e[0] for e in stationInformation],
                   'buildTime': [e[1].strftime('%Y-%m-%d %H:%M:%S') for e in stationInformation]}, 'stationIdOrderByBuildTime.json')