from dataAPI.utils import *


# load the station id on island
clusterLabelsOnIsland = [19, 5, 2, 26, 17, 12, 22, 29, 11, 6]  # 24, 13,
fileName = 'HCLUSTER.json'
clusterResult = getJsonData(fileName)
stationLabelDict = clusterResult['stationLabelDict']
stationIDList = getStationIDList()
stationLabel = [stationLabelDict[e] for e in stationIDList]
# reallocate the cluster name
stationIDListForClustering = [e for e in stationIDList if stationLabelDict[e] in clusterLabelsOnIsland]
transitionMatrix = np.loadtxt(os.path.join(txtPath, 'transitionMatrix.txt'), delimiter=' ')
# get the directed graph
directedGraphWeight = []
demandThreshold = 600
for i in range(len(transitionMatrix)):
    for j in range(len(transitionMatrix[i])):
        if transitionMatrix[i][j] > demandThreshold:
            directedGraphWeight.append([i, j, transitionMatrix[i][j] / np.sum(transitionMatrix, axis=1)[i]])
# get the degree
degreeList = [0 for _ in range(len(stationIDListForClustering))]
for edge in directedGraphWeight:
    start = edge[0]
    end = edge[1]
    weight = edge[2]
    degreeList[start] += 1
    degreeList[end] += 1

# find the largest n degree stations
n = stationIDListForClustering.__len__()
centralStationList = sorted([[e, degreeList[e]] for e in range(n)], key=lambda x:x[1], reverse=True)
for i in range(n, len(degreeList)):
    degree = degreeList[i]
    for j in range(n):
        if centralStationList[j][1] < degree:
            centralStationList.insert(j, [i, degree])
            del centralStationList[-1]
            break
centralStationList = [e[0] for e in centralStationList]
centralStationLocation = [getStationLocation(stationIDListForClustering[e]) for e in centralStationList]
centralStationIDList = [stationIDListForClustering[e] for e in centralStationList]
saveJsonData({'centralStationIDList': centralStationIDList,
              'allStationIDList': stationIDListForClustering}, 'centralStationIDList.json')