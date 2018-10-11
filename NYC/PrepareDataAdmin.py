# master for prepare data

import os

# os.system('python -m PrepareData.WeatherCondition')

# os.system('python -m PrepareData.StationBasicInfo')

# 修改时间段
os.system('python -m PrepareData.GetTransitionMatrix')

# 根据interaction矩阵，使用degree进行排序
os.system('python -m PrepareData.GetCentralStationList')

# os.system('python -m PrepareData.ComputeMinDemand')

os.system('python -m PrepareData.GenerateGraph')

os.system('python -m PrepareData.GetGraphPreDataMulThreads')