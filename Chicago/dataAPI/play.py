import random
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from dataAPI.apis import *

clusterLabelsOnIsland = [19, 5, 2, 26, 17, 12, 22, 29, 11, 6] # 24, 13,

fileName = 'HCLUSTER.json'
clusterResult = getJsonData(fileName)
stationLabelDict = clusterResult['stationLabelDict']
stationLabel = [e[1] for e in stationLabelDict.items() if e[1] in clusterLabelsOnIsland]
stationIDList = [e[0] for e in stationLabelDict.items() if e[1] in clusterLabelsOnIsland]
stationIDListForClustering = [e for e in stationIDList if stationLabelDict[e] in clusterLabelsOnIsland]
stationLabelForClustering = [clusterLabelsOnIsland.index(stationLabelDict[e]) for e in stationIDListForClustering]

ratio = 10.0
stationLocationList = [[e1*ratio for e1 in getStationLocation(e)] for e in stationIDListForClustering]
transitionMatrix = np.loadtxt(os.path.join(txtPath, 'transitionMatrix.txt'), delimiter=' ')
rideRecords = []
recordMinThreshold = 100
plotList = []
for i in range(stationIDListForClustering.__len__()):
    for j in range(stationIDListForClustering.__len__()):
        if transitionMatrix[i][j] > recordMinThreshold:
            rideRecords.append([getStationLocation(stationIDListForClustering[i]),
                                getStationLocation(stationIDListForClustering[j])])
            plotList.append([stationLocationList[i], stationLocationList[j]])

for i in range(plotList.__len__()):
    print(i)
    plt.plot(plotList[i][0], plotList[i][1])
plt.show()