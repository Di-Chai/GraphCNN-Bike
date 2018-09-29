# a "jsonPath" variable need to be provided for this file (for function getJsonData and saveJsonData)
import json
import os
from localPath import jsonPath

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
    
def getJsonDataFromPath(fullPath, showMessage=False):
    with open(fullPath, 'r') as f:
        data = json.load(f)
    if showMessage:
        print('load', fullPath)
    return data


def saveJsonDataToPath(dataDict, fullPath, showMessage=True):
    with open(fullPath, 'w') as f:
        json.dump(dataDict, f)
    if showMessage:
        print('Saved', fullPath)
