from localPath import *
import os
import csv
import json
import numpy as np
from dateutil.parser import parse
from Utils.symbols import *
dateTimeMode = '%Y-%m-%d'

def getDateFromTimeString(timeString):
    return parse(timeString).strftime(dateTimeMode)

isBadDay = {}
temperature = {}
temperatureHour = {}
wind = {}
windHour = {}
with open(os.path.join(csvDataPath, '1135040.csv'), 'r') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    counter = 0
    for row in f_csv:
        print(counter)
        counter += 1
        # date 05
        currentDate = row[5]
        date = parse(currentDate)
        dateString = date.strftime(dateTimeMode)
        hour = date.hour
        # 天气情况
        if dateString not in isBadDay:
            isBadDay[dateString] = 0
        weatherCondition = row[9]
        if weatherCondition != '':
            isBadDay[dateString] = 1
        # 气温
        if dateString not in temperature:
            temperature[dateString] = []
            temperatureHour[dateString] = [NO_TEM for e in range(24)]
        if row[15] != '':
            try:
                temperature[dateString].append(float(row[15].replace('s', '')))
                temperatureHour[dateString][hour] = float(row[15].replace('s', ''))
            except:
                print('Tem', row[15])
        #  风速
        if dateString not in wind:
            wind[dateString] = []
            windHour[dateString] = [0 for e in range(24)]
        if row[17] != '':
            try:
                wind[dateString].append(float(row[17].replace('s', '')))
                windHour[dateString][hour] = float(row[17].replace('s', ''))
            except:
                print('Wind', row[17])

for key, value in temperature.items():
    temperature[key] = NO_TEM if temperature[key].__len__() == 0 else np.mean(value)
for key, value in wind.items():
    wind[key] = NO_TEM if wind[key].__len__() == 0 else np.mean(value)

weatherDict = {'isBadDay': isBadDay, 'temperature': temperature, 'wind': wind, 'temHour': temperatureHour,
               'windHour': windHour}

with open(os.path.join(jsonPath, 'weatherDictChicago.json'), 'w') as f:
    json.dump(weatherDict, f)

