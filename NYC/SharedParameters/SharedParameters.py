n_central_stations = 424
timeSlot = 1
timeSlotV2 = 60
# (0) feature and target length
featureLength = 6
targetLength = 1
lossTestLength = 100
pValueConfidenceValue = 0.1

# (1) LSTM AutoEncoder Parameters
n_epoch = 5001
batch_size = 128
lr = 0.0001  # learning rate
n_hidden_units = 64
dropout_pro = 0.05
n_inputs = 1
n_steps_encoder = featureLength
n_steps_decoder = 3
decoderOutputSize = 1

timeRange = ['2013-07-01', '2017-09-30']
trainDataTimeRange = ['2013-07-01', '2016-09-30']
# valDataTimeRange = ['2016-10-01', '2016-12-31']
# testDataRange = ['2017-01-01', '2017-09-30']

# trainDataLength = 186 # 22 60
valDataLength = 40
testDataLength = 80

demandTypeList = ['in', 'out', 'sum']
demandType = demandTypeList[0]