import os

# raw bike data and processing
# -> basic bike information
# -> traffic flow data

# raw weather data and processing

os.system('python getTransitionMatrix.py')

os.system('python generateGraph.py')

os.system('python getCentralStationList.py')

os.system('python getGraphPreDataMulThreadsV2.py')

os.system('python copyFile.py')

os.system('python main.py')