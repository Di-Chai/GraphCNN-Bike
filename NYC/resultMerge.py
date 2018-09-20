from localPath import *
import numpy as np

finalPre_ = []
testTarget = []

codeVersion = 'GraphFusionModelV10'
targetFolder = GraphDemandPreDataPath
txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt') and e.startswith(codeVersion)]
txtFileRankList = [int(e.split('-')[1]) for e in txtFileName]
txtFileName = [e[1] for e in sorted(zip(txtFileRankList, txtFileName), key = lambda x: x[0])]

finalPreResultFileList = [txtFileName[e] for e in range(len(txtFileName)) if 'finalPreResult' in txtFileName[e]]
testTargetFileList = [txtFileName[e] for e in range(len(txtFileName)) if 'testTarget' in txtFileName[e]]

finalPre0 = []
for file in finalPreResultFileList:
    tmp = np.loadtxt(os.path.join(GraphDemandPreDataPath, file), delimiter=' ')
    finalPre0.append(tmp)

for file in testTargetFileList:
    tmp = np.loadtxt(os.path.join(GraphDemandPreDataPath, file), delimiter=' ')
    testTarget.append(tmp)

codeVersion = 'GraphSingleStationDemandPreV3'
targetFolder = GraphDemandPreDataPath
txtFileName = [e for e in os.listdir(targetFolder) if e.endswith('.txt') and e.startswith(codeVersion)]
txtFileRankList = [int(e.split('-')[1]) for e in txtFileName]
txtFileName = [e[1] for e in sorted(zip(txtFileRankList, txtFileName), key = lambda x: x[0])]

finalPreResultFileList = [txtFileName[e] for e in range(len(txtFileName)) if 'finalPreResult' in txtFileName[e]]
testTargetFileList = [txtFileName[e] for e in range(len(txtFileName)) if 'testTarget' in txtFileName[e]]

finalPre1 = []
finalPre2 = []
for i in range(len(finalPreResultFileList)):
    if i % 3 == 0:
        continue
    file = finalPreResultFileList[i]
    tmp = np.loadtxt(os.path.join(GraphDemandPreDataPath, file), delimiter=' ')
    if i % 3 == 1:
        finalPre1.append(tmp)
    if i % 3 == 2:
        finalPre2.append(tmp)

finalPre_.append(finalPre0)
finalPre_.append(finalPre1)
finalPre_.append(finalPre2)

finalPre_ = np.array(finalPre_, dtype=np.float32)

aggFinalPre = np.average(finalPre_, axis=0)

testTarget = np.array(testTarget, dtype=np.float32)

finalPre = np.mean(finalPre_, axis=0)

for i in range(len(finalPre)):
    modelName = 'GraphFusionModelV99'
    np.savetxt(os.path.join(GraphDemandPreDataPath, modelName + '-%s' % i + '-finalPreResult.txt'),
               finalPre[i],
               newline='\n', delimiter=' ')
    np.savetxt(os.path.join(GraphDemandPreDataPath, modelName + '-%s' % i + '-testTarget.txt'),
               testTarget[i],
               newline='\n', delimiter=' ')