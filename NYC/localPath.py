import os

projectPath = os.path.dirname(os.path.abspath(__file__))

dataPath = os.path.join(os.path.join(os.path.dirname(projectPath), 'DataDir'), 'NYC')

csvDataPath = os.path.join(dataPath, 'csvData')

jsonPath = os.path.join(dataPath, 'json')

demandDataPath = os.path.join(dataPath, 'demandData')

pngPath = os.path.join(dataPath, 'png')

rawBikeDataPath = os.path.join(dataPath, 'RawBikeData')

demandMinDataPath = os.path.join(dataPath, 'demandMinData')

txtPath = os.path.join(dataPath, 'TXT')

clusterFilePath = os.path.join(dataPath, 'clusterFilePath')

GraphDemandPreDataPath = os.path.join(dataPath, 'GraphDemandPreData')

if __name__ == '__main__':
    dirList = [projectPath, dataPath, csvDataPath, jsonPath, demandDataPath, pngPath, rawBikeDataPath,
            demandMinDataPath, txtPath, GraphDemandPreDataPath]
    for dir in dirList:
        if os.path.isdir(dir) == False:
            os.mkdir(dir)