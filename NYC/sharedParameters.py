n_central_stations = 10
timeSlot = 1
timeSlotV2 = 60
# (0) feature and target length
featureLength = 180
targetTimeSlot = 60
targetLength = 1
lossTestLength = 40
pValueConfidenceValue = 0.1

# (1) LSTM AutoEncoder Parameters
n_epoch = 5000
batch_size = 128
lr = 0.0001  # learning rate
n_hidden_units = 64
dropout_pro = 0.05
n_inputs = 1
n_steps_encoder = featureLength
n_steps_decoder = int(featureLength / targetTimeSlot)
decoderOutputSize = 1

timeRange = ['2016-01-01', '2017-09-30']
trainDataTimeRange = ['2016-01-01', '2017-03-01']
valDataTimeRange = ['2017-03-01', '2017-05-01']
testDataRange = ['2017-05-01', '2017-09-30']

# trainDataLength = 186 # 22 60
valDataLength = 20
testDataLength = 40

demandTypeList = ['in', 'out', 'sum']
demandType = demandTypeList[0]