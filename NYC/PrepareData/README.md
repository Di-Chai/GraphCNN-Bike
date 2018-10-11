# Code excute order

#### The Code should be excuted in the following order : 

(1) Process weather data

```
python -m PrepareData.WeatherCondition
```

(2) Access basic information like build-time, location and so on

```
python -m PrepareData.StationBasicInfo
```

(3) Get the transition matrix, which will be used for calculating central stations and building interation graph

```
python -m PrepareData.GetTransitionMatrix
```

(4) Order the stations based on degree (use transition matrix)

```
python -m PrepareData.GetCentralStationList
```

(5) Data preprocessing, get demand data (This step need a large RAM, as least 32 GB)

```
python -m PrepareData.ComputeMinDemand
```

(6) Generate all the Graphs

```
python -m PrepareData.GenerateGraph
```

(7) Prepare the graph-structure data, which will be used in the GCN part.

```
python -m PrepareData.GetGraphPreDataMulThreads
```

