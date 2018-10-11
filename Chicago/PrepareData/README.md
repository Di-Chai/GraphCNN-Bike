# 代码执行顺序

处理天气数据
1 python -m PrepareData.WeatherCondition

获取站点新建时间、经纬度信息
2 python -m PrepareData.StationBasicInfo

获取transition矩阵，用于构建Graph
站点按构建时间排序
3 python -m PrepareData.GetTransitionMatrix

按照Degree排序，得到central station list
4 python -m PrepareData.GetCentralStationList

计算每分钟的demand数据
和站点的排序无关
5 python -m PrepareData.ComputeMinDemand

生成所有的graph
和站点的排序有关，需要注意！！
6 python -m PrepareData.GenerateGraph

生成 graph cnn 所需的图
7 python -m PrepareData.GetGraphPreDataMulThreads