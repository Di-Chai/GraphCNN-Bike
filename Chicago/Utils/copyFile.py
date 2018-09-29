import os

def copyFileFromTo(fileStart, target, destList):
    currentPath = os.path.dirname(os.path.abspath(__file__))
    allFileList = [e for e in os.listdir(currentPath) if e.startswith(fileStart) and e.endswith('.py')]
    allFileName = [e[:-3] for e in allFileList]

    rankRange = destList
    sourceRank = target

    sourceFile = ''
    targetFileList = []
    for i in range(allFileName.__len__()):
        fileName = allFileName[i].split('_')
        rank = int(fileName[-1])
        if rank == sourceRank:
            sourceFile = allFileList[i]
        elif rank >= min(rankRange) and rank <= max(rankRange):
            targetFileList.append(allFileList[i])

    # get source file content
    if sourceFile == '':
        print('Can not find source file')
    else:
        with open(os.path.join(currentPath, sourceFile), 'r', encoding='utf-8') as f:
            sourceFileContent = f.readlines()
        if targetFileList.__len__() == 0:
            print('No target file')
        else:
            for targetFile in targetFileList:
                with open(os.path.join(currentPath, targetFile), 'w', encoding='utf-8') as f:
                    f.writelines(sourceFileContent)
    print('Succeed')

def generateFile(fileNameString, target, destList):
    currentPath = os.path.dirname(os.path.abspath(__file__))
    sourceFile = fileNameString % target
    targetFileList = []
    for fileCounter in range(destList[0], destList[1] + 1):
        targetFileList.append(fileNameString % fileCounter)

    # get source file content
    if sourceFile == '':
        print('Can not find source file')
    else:
        with open(os.path.join(currentPath, sourceFile), 'r', encoding='utf-8') as f:
            sourceFileContent = f.readlines()
        if targetFileList.__len__() == 0:
            print('No target file')
        else:
            for targetFile in targetFileList:
                with open(os.path.join(currentPath, targetFile), 'w', encoding='utf-8') as f:
                    f.writelines(sourceFileContent)
    print('Succeed')

if __name__ == '__main__':
    # copyFileFromTo('GraphSingleStationDemandPre', 0, [1, 9])
    copyFileFromTo('GraphSingleStationDemandPreV2', 0, [1, 299])
    copyFileFromTo('GraphFusionModel_', 0, [1, 299])
    # generateFile('GraphFusionModel_%s.py', 0, [1, 299])
