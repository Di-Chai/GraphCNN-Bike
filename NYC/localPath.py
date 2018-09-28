import os

projectPath = os.path.dirname(os.path.abspath(__file__))

dataPath = os.path.join(os.path.join(os.path.dirname(projectPath), 'DataDir'), 'NYC')

csvDataPath = os.path.join(dataPath, 'csvData')

htmlPath = os.path.join(dataPath, 'html')

jsonPath = os.path.join(dataPath, 'json')

demandDataPath = os.path.join(dataPath, 'demandData')

pngPath = os.path.join(dataPath, 'png')

rawBikeDataPath = os.path.join(dataPath, 'RawBikeData')

predictionPngPath = os.path.join(pngPath, 'Prediction')

demandMinDataPath = os.path.join(dataPath, 'demandMinData')

tfModelDataPath = os.path.join(dataPath, 'tfModelDataPath')

txtPath = os.path.join(dataPath, 'TXT')

in_demand_pre_pngPath = os.path.join(pngPath, 'in-demand-pre')

clusterDemandPrePng = os.path.join(pngPath, 'clusterDemandPre')

clusterFilePath = os.path.join(dataPath, 'clusterFilePath')

clusterDemandPreDataPath = os.path.join(dataPath, 'clusterDemandPreDataPath')

GraphDemandPreDataPath = os.path.join(dataPath, 'GraphDemandPreData')

if __name__ == '__main__':
    dirList = [projectPath, dataPath, csvDataPath, htmlPath, jsonPath, demandDataPath, pngPath, rawBikeDataPath, predictionPngPath,
            demandMinDataPath, tfModelDataPath, txtPath, in_demand_pre_pngPath, clusterDemandPreDataPath, GraphDemandPreDataPath]
    for dir in dirList:
        if os.path.isdir(dir) == False:
            os.mkdir(dir)