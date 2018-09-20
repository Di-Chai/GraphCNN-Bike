from dataAPI.utils import *
import tensorflow as tf
from scipy import stats
import sys

argvList = sys.argv
currentFileName = argvList[1]
codeVersion = currentFileName.replace('_', '-')
rank = int(currentFileName.split('_')[-1])

stationIDDict = getJsonData('centralStationIDList.json')
centralStationIDList = stationIDDict['centralStationIDList']
allStationIDList = stationIDDict['allStationIDList']

# (0) feature and target length
featureLength = 6
targetLength = 1
lossTestLength = 300
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

# date = parse(trainDataTimeRange[0])
# dayCounter = 0
# while date <= parse(trainDataTimeRange[1]):
#     dateString = date.strftime(dateTimeMode)
#     if isWorkDay(dateString) and isBadDay(dateString) == False:
#         dayCounter += 1
#     date = date + datetime.timedelta(days=1)
#
# print(dayCounter)

# trainDataLength = 186 # 22 60
valDataLength = 40
testDataLength = 80

demandTypeList = ['in', 'out', 'sum']
demandType = demandTypeList[0]

# (2) prediction network
# with the first layer and the last layer
n_epoch_pre = 5001
addFeatureLength = 2 # additionalFeatureLengthDict[str(rank)]
predictLength = targetLength
n_units_pre = [n_hidden_units + addFeatureLength + 5, n_hidden_units*2, n_hidden_units, n_hidden_units/2, predictLength]
n_units_pre = [int(e) for e in n_units_pre]

# (3) uncertainty parameters
varB = 100

# (4) parameter for saving the model
AutoEncoderModelFileName = 'AutoEncoderModel-%s' % (codeVersion)
autoEncoderModelFileSavePath = os.path.join(GraphDemandPreDataPath, AutoEncoderModelFileName)
print(autoEncoderModelFileSavePath)
if not os.path.exists(autoEncoderModelFileSavePath):
    os.makedirs(autoEncoderModelFileSavePath)
preModelFileName = 'PredModel-%s' % (codeVersion)
preModelFileSavePath = os.path.join(GraphDemandPreDataPath, preModelFileName)
if not os.path.exists(preModelFileSavePath):
    os.makedirs(preModelFileSavePath)

saveSteps = 100

autoEncoderFileExist = False
preFileExist = False
if os.listdir(autoEncoderModelFileSavePath).__len__() > 0:
    autoEncoderFileExist = True
if os.listdir(preModelFileSavePath).__len__() > 0:
    preFileExist = True

trainAutoEncoder = True
trainPreModel = True

######################################################################################################################
# Prepare the training data and test data
######################################################################################################################
if os.path.isfile(os.path.join(jsonPath, 'GraphPreData.json')) is False:
    os.system('python getGraphPreDataMulThreads.py')

GraphValueData = getJsonData('GraphValueMatrix.json')

GraphValueMatrix = GraphValueData['GraphValueMatrix']
tem = GraphValueData['tem']
wind = GraphValueData['wind']

trainDataLength = len(GraphValueMatrix) - valDataLength - testDataLength

allTrainData = GraphValueMatrix[0: trainDataLength]
trainTemList = tem[0: trainDataLength]
trainWindList = wind[0: trainDataLength]

allValData = GraphValueMatrix[trainDataLength: trainDataLength + valDataLength]
valTemList = tem[trainDataLength: trainDataLength + valDataLength]
valWindList = wind[trainDataLength: trainDataLength + valDataLength]

allTestData = GraphValueMatrix[-testDataLength:]
testTemList = tem[-testDataLength:]
testWindList = wind[-testDataLength:]

del GraphValueData
del GraphValueMatrix
del tem
del wind

def moveSample(demandData, temData, windData):
    Feature0 = []
    Target0 = []
    for j in range(len(demandData)):
        dailyRecord = demandData[j]
        for k in range(len(dailyRecord) - featureLength - targetLength + 1):
            Feature0.append([dailyRecord[k: k + featureLength],
                                  [temData[len(temData) - len(demandData) + j][k],
                                   windData[len(windData) - len(demandData) + j][k],
                                  ],
                                  ])
            Target0.append(dailyRecord[k + featureLength: k + featureLength + targetLength])
    return Feature0, Target0

# one hour
trainFeature0, trainTarget0 = moveSample(allTrainData, trainTemList, trainWindList)
testFeature0, testTarget0 = moveSample(allTestData, testTemList, testWindList)
valFeature0, valTarget0 = moveSample(allValData, valTemList, valWindList)

del allTrainData
del allTestData
del allValData
del trainTemList
del testTemList
del valTemList
del trainWindList
del testWindList
del valWindList

autoEncoderFeature = [e[0] for e in trainFeature0] + [e[0] for e in valFeature0]
autoEncoderTarget = trainTarget0 + valTarget0

preFeatuer = [e[0] for e in trainFeature0]
preTarget = trainTarget0
preOtherFeature = [e[1] for e in trainFeature0]

valFeature = [e[0] for e in valFeature0]
valTarget = valTarget0
valOtherFeature =  [e[1] for e in valFeature0]

testFeature = [e[0] for e in testFeature0]
testTarget = testTarget0
testOtherFeature =  [e[1] for e in testFeature0]

# clear the cache
del trainFeature0
del trainTarget0
del testFeature0
del testTarget0
del valFeature0
del valTarget0

autoEncoderFeature = np.array(autoEncoderFeature, dtype=np.float32)
autoEncoderTarget = np.array(autoEncoderTarget, dtype=np.float32)
preFeatuer = np.array(preFeatuer, dtype=np.float32)
preTarget = np.array(preTarget, dtype=np.float32)
preOtherFeature = np.array(preOtherFeature, dtype=np.float32)
valFeature = np.array(valFeature, dtype=np.float32)
valTarget = np.array(valTarget, dtype=np.float32)
valOtherFeature = np.array(valOtherFeature, dtype=np.float32)
testFeature = np.array(testFeature, dtype=np.float32)
testTarget = np.array(testTarget, dtype=np.float32)
testOtherFeature = np.array(testOtherFeature, dtype=np.float32)

stationNumber = autoEncoderFeature.shape[2]
targetStationIndex = rank * 4
print('Station', targetStationIndex)

distanceGraphMatrix = np.loadtxt(os.path.join(txtPath, 'distanceGraphMatrix.txt'), delimiter=' ')
demandGraphMatrix = np.loadtxt(os.path.join(txtPath, 'demandGraphMatrix.txt'), delimiter=' ')
demandMask = np.loadtxt(os.path.join(txtPath, 'demandMask.txt'), delimiter=' ')
fusionGraph = np.loadtxt(os.path.join(txtPath, 'fusionGraphMatrix.txt'), delimiter=' ')

demandMaskTensorFeed = np.array(demandMask[targetStationIndex]).reshape([stationNumber, 1])
distanceGraphMatrixFeed = np.array(distanceGraphMatrix[allStationIDList.index(centralStationIDList[targetStationIndex])])\
                            .reshape([stationNumber, 1])
demandGraphMatrixFeed = np.array(demandGraphMatrix[allStationIDList.index(centralStationIDList[targetStationIndex])]) \
                          .reshape([stationNumber, 1])
fusionGraphMatrixFeed = np.array(fusionGraph[allStationIDList.index(centralStationIDList[targetStationIndex])]) \
                          .reshape([stationNumber, 1])
###################################################################################################################
# Build The NetWork
#########################################################################
# Define Graphs
autoEncoderGraph = tf.Graph()
preNNGraph = tf.Graph()

with autoEncoderGraph.as_default():

    # LSTM AutoEncoder
    encoderRawInput = tf.placeholder(tf.float32, [batch_size, n_steps_encoder, stationNumber])
    decoderRawInput = tf.placeholder(tf.float32, [batch_size, n_steps_decoder, stationNumber])
    decoderRawTarget = tf.placeholder(tf.float32, [batch_size, stationNumber])
    demandMaskTensorAutoEncoder = tf.placeholder(tf.float32, [stationNumber, 1])

    with tf.variable_scope('encoder_fusionGraph'):
        # channel 1
        fusionGraphMatrixTensorRaw = tf.placeholder(tf.float32, [stationNumber, 1])
        fusionGraphMatrixWeight = tf.matrix_diag(tf.Variable(tf.random_normal([stationNumber])))
        fusionGraphMatrixTensor = tf.matmul(fusionGraphMatrixWeight, fusionGraphMatrixTensorRaw)

        encoderCell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, state_is_tuple=True)
        encoderCell = tf.nn.rnn_cell.DropoutWrapper(encoderCell, output_keep_prob=1 - dropout_pro)

        encoderInput = tf.reshape(
            tf.matmul(tf.reshape(encoderRawInput, [-1, stationNumber]), fusionGraphMatrixTensor),
            [-1, n_steps_encoder, 1])

        en_outputs, en_final_state = tf.nn.dynamic_rnn(encoderCell, encoderInput, dtype=tf.float32, time_major=False)

    # Decoder
    with tf.variable_scope('decoder') as vs:
        decoderInput = tf.reshape(
            tf.matmul(tf.reshape(decoderRawInput, [-1, stationNumber]), demandMaskTensorAutoEncoder),
            [batch_size, n_steps_decoder, 1])
        decoderTarget = tf.matmul(decoderRawTarget, demandMaskTensorAutoEncoder)
        decoderCell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, state_is_tuple=True)
        decoderCell = tf.nn.rnn_cell.DropoutWrapper(decoderCell, output_keep_prob=1 - dropout_pro)
        decoderOutputUnitNumber = n_hidden_units
        decoderOutputWeight = tf.Variable(tf.random_normal([decoderOutputUnitNumber, n_inputs]), name='decoderOutputWeight')
        decoderOutputBias = tf.Variable(tf.constant(0.1, shape=[n_inputs, ]), name='encoderInputBias')
        decoderState = en_final_state
        de_outputs, de_final_states = tf.nn.dynamic_rnn(decoderCell, decoderInput, initial_state=decoderState,
                                                        time_major=False)
        de_outputs = tf.reshape(tf.matmul(tf.reshape(de_outputs[:, -1, -decoderOutputUnitNumber:], [-1, decoderOutputUnitNumber]),
                      decoderOutputWeight) + decoderOutputBias, [-1, n_inputs])
    loss_autoEncoder = tf.reduce_mean(tf.square(decoderTarget - de_outputs))
    trainOperation_autoEncoder = tf.train.AdamOptimizer(lr).minimize(loss_autoEncoder)

    autoEncoderSaver = tf.train.Saver(max_to_keep=None)
    init_autoEncoder = tf.global_variables_initializer()

# prediction network
# fully connected
with preNNGraph.as_default():
    demandMaskTensorPreNN = tf.placeholder(tf.float32, [stationNumber, 1])
    inputEmbeddingFeature = tf.placeholder(tf.float32, [None, n_hidden_units])
    preNNInputRaw = tf.placeholder(tf.float32, [batch_size, stationNumber])
    preNNTargetRaw = tf.placeholder(tf.float32, [batch_size, stationNumber])
    preNNTarget = tf.matmul(preNNTargetRaw, demandMaskTensorPreNN)
    preNNInput = tf.matmul(preNNInputRaw, demandMaskTensorPreNN)
    otherFeature = tf.placeholder(tf.float32, [None, addFeatureLength])
    inputFeature = tf.concat([inputEmbeddingFeature, otherFeature,
                              preNNInput, preNNInput, preNNInput, preNNInput, preNNInput], 1)
    with tf.variable_scope('predict') as vs_p:
        W = [tf.Variable(tf.random_normal([n_units_pre[e], n_units_pre[e + 1]])) for e in range(n_units_pre.__len__() - 1)]
        B = [tf.Variable(tf.constant(0.1, shape=[n_units_pre[e], ])) for e in range(1, n_units_pre.__len__())]
        preOutput = inputFeature
        # remove the last layer (for output - linearRegression)
        for i in range(n_units_pre.__len__() - 2):
            vs_p.reuse_variables()
            preOutput = tf.nn.tanh(tf.matmul(preOutput, W[i]) + B[i])
            preOutput = tf.nn.dropout(preOutput, keep_prob=1 - dropout_pro)
        preOutput = tf.matmul(preOutput, W[-1]) + B[-1]
        # tv = vs_p.trainable_variables()
        # regularization_cost = 0.0001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    loss_pre = tf.reduce_mean(tf.square(preNNTarget - preOutput)) # + regularization_cost
    trainOperation_pre = tf.train.AdamOptimizer(lr).minimize(loss_pre)

    preSaver = tf.train.Saver(max_to_keep=None)
    init_preNN = tf.global_variables_initializer()

#########################################################################
# Train The Network
#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True

autoEncoderSession = tf.Session(config=config, graph=autoEncoderGraph)

autoEncoderSession.run(init_autoEncoder)
# train the autoEncoder
autoEncoderLossListEpoch = []
finishedEpoch = 1
if trainAutoEncoder:
    if autoEncoderFileExist:
        modelFiles = [e for e in os.listdir(autoEncoderModelFileSavePath)]
        modelFileNumbers = []
        for e in modelFiles:
            try:
                modelFileNumbers.append(int(e))
            except:
                print(e)
        currentFileSavePath = os.path.join(autoEncoderModelFileSavePath, str(max(modelFileNumbers)))
        autoEncoderSaver.restore(autoEncoderSession, os.path.join(currentFileSavePath,AutoEncoderModelFileName))
        finishedEpoch += int(max(modelFileNumbers))
        autoEncoderLossListEpoch = getJsonDataFromPath(os.path.join(autoEncoderModelFileSavePath, 'autoEncoderLoss.json'))['autoEncoderLossList']
        autoEncoderLossListEpoch = [float(e) for e in autoEncoderLossListEpoch]
    for epoch in range(finishedEpoch, n_epoch):
        currentIterations = int(len(autoEncoderFeature) / batch_size)
        if len(autoEncoderFeature) % batch_size != 0:
            currentIterations += 1
        trainLossList = []
        for iteration in range(currentIterations):
            pointer0 = iteration * batch_size
            if len(autoEncoderFeature) < pointer0 + batch_size:
                pointer1 = len(autoEncoderFeature)
                pointer0 = pointer1 - batch_size
                pointer0 = max(0, pointer0)
            else:
                pointer1 = pointer0 + batch_size
            batch_encoder_input = autoEncoderFeature[pointer0:pointer1].reshape([batch_size, n_steps_encoder, stationNumber])
            batch_decoder_input = autoEncoderFeature[pointer0:pointer1][:, -n_steps_decoder:, :].reshape([batch_size, n_steps_decoder, stationNumber])
            batch_decoder_target = autoEncoderTarget[pointer0: pointer1].reshape([batch_size, stationNumber])
            trainLoss, _ = autoEncoderSession.run([loss_autoEncoder, trainOperation_autoEncoder],
                                    feed_dict={
                                        demandMaskTensorAutoEncoder: demandMaskTensorFeed,
                                        fusionGraphMatrixTensorRaw: fusionGraphMatrixFeed,
                                        encoderRawInput: batch_encoder_input,
                                        decoderRawInput: batch_decoder_input,
                                        decoderRawTarget: batch_decoder_target
                                    })
            trainLossList.append(trainLoss)
        print('%s Training loss_autoEncoder after %s epoch : %s' % (codeVersion, epoch, np.mean(trainLossList)))
        autoEncoderLossListEpoch.append(np.mean(trainLossList))
        if epoch >= (lossTestLength * 2):
            lossTTest = stats.ttest_ind(autoEncoderLossListEpoch[-lossTestLength:],
                                        autoEncoderLossListEpoch[-lossTestLength * 2:-lossTestLength],
                                        equal_var=False)
            ttest = lossTTest[0]
            pValue = lossTTest[1]
            print('ttest:', ttest, 'pValue', pValue)
            if pValue > pValueConfidenceValue or ttest > 0:
                break
        if epoch % saveSteps == 0:
            currentSavePath = os.path.join(autoEncoderModelFileSavePath, str(epoch))
            if not os.path.exists(currentSavePath):
                os.makedirs(currentSavePath)
            autoEncoderSaver.save(sess=autoEncoderSession, save_path=os.path.join(currentSavePath, AutoEncoderModelFileName))
            saveJsonDataToPath({
                'autoEncoderLossList': [str(e) for e in autoEncoderLossListEpoch]
            }, os.path.join(autoEncoderModelFileSavePath, 'autoEncoderLoss.json'))
else:
    modelFiles = [e for e in os.listdir(autoEncoderModelFileSavePath) if e.split('.').__len__() == 1]
    modelFileNumbers = []
    for e in modelFiles:
        try:
            modelFileNumbers.append(int(e))
        except:
            print(e)
    currentSavePath = os.path.join(autoEncoderModelFileSavePath, str(max(modelFileNumbers)))
    autoEncoderSaver.restore(autoEncoderSession, os.path.join(currentSavePath, AutoEncoderModelFileName))

preNNSession = tf.Session(config=config, graph=preNNGraph)

preNNSession.run(init_preNN)
if trainPreModel:
    finishedEpoch = 1
    preLossListEpoch = []
    if preFileExist:
        modelFiles = [e for e in os.listdir(preModelFileSavePath)]
        modelFileNumbers = []
        for e in modelFiles:
            try:
                modelFileNumbers.append(int(e))
            except:
                print(e)
        currentFileSavePath = os.path.join(preModelFileSavePath, str(max(modelFileNumbers)))
        autoEncoderSaver.restore(preNNSession, os.path.join(currentFileSavePath, preModelFileName))
        finishedEpoch += int(max(modelFileNumbers))
        preLossListEpoch = getJsonDataFromPath(os.path.join(preModelFileSavePath, 'preLoss.json'))['preLossList']
        preLossListEpoch = [float(e) for e in preLossListEpoch]
    for epoch in range(finishedEpoch, n_epoch_pre):
        preLossList = []
        currentIterations = int(len(preFeatuer) / batch_size)
        if len(preFeatuer) % batch_size != 0:
            currentIterations += 1
        for iteration in range(currentIterations):
            pointer0 = iteration * batch_size
            if len(preFeatuer) < pointer0 + batch_size:
                pointer1 = len(preFeatuer)
                pointer0 = pointer1 - batch_size
                pointer0 = max(0, pointer0)
            else:
                pointer1 = pointer0 + batch_size
            batch_encoder_input = preFeatuer[pointer0:pointer1].reshape([batch_size, n_steps_encoder, stationNumber])
            batchOtherFeature = preOtherFeature[pointer0:pointer1].reshape([batch_size, addFeatureLength])
            batch_preNN_input = preFeatuer[pointer0: pointer1][:, -1, :].reshape([batch_size, stationNumber])
            batch_preNN_target = preTarget[pointer0: pointer1].reshape([batch_size, stationNumber])
            finalState = autoEncoderSession.run(
                en_final_state,
                feed_dict={
                    demandMaskTensorAutoEncoder: demandMaskTensorFeed,
                    fusionGraphMatrixTensorRaw: fusionGraphMatrixFeed,
                    encoderRawInput: batch_encoder_input,
                }
            )
            lossPre, _, output = preNNSession.run([loss_pre, trainOperation_pre, preOutput],
                                          feed_dict={
                                              demandMaskTensorPreNN: demandMaskTensorFeed,
                                              # H
                                              inputEmbeddingFeature: finalState[-1],
                                              otherFeature: batchOtherFeature,
                                              # C, maybe c is better
                                              #inputEmbeddingFeature: finalState[0],
                                              preNNTargetRaw: batch_preNN_target,
                                              preNNInputRaw: batch_preNN_input
                                          })
            preLossList.append(lossPre)
        print('%s Train loss_pre at epoch %s: %s' % (codeVersion, epoch, np.mean(preLossList)))
        preLossListEpoch.append(np.mean(preLossList))
        if epoch >= (lossTestLength * 2):
            lossTTest = stats.ttest_ind(preLossListEpoch[-lossTestLength:],
                                        preLossListEpoch[-lossTestLength * 2:-lossTestLength], equal_var=False)
            ttest = lossTTest[0]
            pValue = lossTTest[1]
            print('ttest:', ttest, 'pValue', pValue)
            if pValue > pValueConfidenceValue or ttest > 0:
                break
        if epoch % saveSteps == 0:
            currentSavePath = os.path.join(preModelFileSavePath, str(epoch))
            if not os.path.exists(currentSavePath):
                os.makedirs(currentSavePath)
            preSaver.save(sess=preNNSession,save_path=os.path.join(currentSavePath, preModelFileName))
            saveJsonDataToPath({
                'preLossList': [str(e) for e in preLossListEpoch]
            }, os.path.join(preModelFileSavePath, 'preLoss.json'))
else:
    modelFiles = [e for e in os.listdir(preModelFileSavePath) if e.split('.').__len__() == 1]
    modelFileNumbers = []
    for e in modelFiles:
        try:
            modelFileNumbers.append(int(e))
        except:
            print(e)
    currentSavePath = os.path.join(preModelFileSavePath, str(max(modelFileNumbers)))
    preSaver.restore(preNNSession, os.path.join(currentSavePath, preModelFileName))

# get the prediction and uncertainty
mulPredictionList = []
for varIter0 in range(varB):
    print(codeVersion, 'varIter0 : ', varIter0)
    tmpPreList = []
    currentIterations = int(len(testFeature) / batch_size)
    if len(testFeature) % batch_size != 0:
        currentIterations += 1
    for iteration in range(currentIterations):
        pointer0 = iteration * batch_size
        if len(testFeature) < pointer0 + batch_size:
            pointer1 = len(testFeature)
            actLen = pointer1 - pointer0
            pointer0 = pointer1 - batch_size
            pointer0 = max(0, pointer0)
        else:
            pointer1 = pointer0 + batch_size
            actLen = batch_size
        batch_encoder_input = testFeature[pointer0:pointer1].reshape(
            [batch_size, n_steps_encoder, stationNumber])
        batchOtherFeature = testOtherFeature[pointer0:pointer1].reshape([batch_size, addFeatureLength])
        batch_preNN_input = testFeature[pointer0: pointer1][:, -1, :].reshape([batch_size, stationNumber])
        batch_preNN_target = testTarget[pointer0: pointer1].reshape([batch_size, stationNumber])
        finalState = autoEncoderSession.run(
            en_final_state,
            feed_dict={
                demandMaskTensorAutoEncoder: demandMaskTensorFeed,
                fusionGraphMatrixTensorRaw: fusionGraphMatrixFeed,
                encoderRawInput: batch_encoder_input,
            }
        )
        prediction = preNNSession.run(preOutput,
                              feed_dict={
                                  demandMaskTensorPreNN: demandMaskTensorFeed,
                                  # H
                                  inputEmbeddingFeature: finalState[-1],
                                  otherFeature: batchOtherFeature,
                                  # C, maybe c is better
                                  # inputEmbeddingFeature: finalState[0],
                                  preNNTargetRaw: batch_preNN_target,
                                  preNNInputRaw: batch_preNN_input
                              })
        if iteration == 0:
            tmpPreList = prediction[batch_size - actLen:batch_size]
        else:
            tmpPreList = np.concatenate((tmpPreList, prediction[batch_size - actLen:batch_size]), axis=0)
    mulPredictionList.append(tmpPreList)
mulPredictionList = np.array(mulPredictionList, dtype=np.float32)

valPredictionList = []
currentIterations = int(len(valFeature) / batch_size)
if len(valFeature) % batch_size != 0:
    currentIterations += 1
for varIter1 in range(varB):
    print(codeVersion, 'varIter1 : ', varIter1)
    tmpPreList = []
    for iteration in range(currentIterations):
        pointer0 = iteration * batch_size
        if len(valFeature) < pointer0 + batch_size:
            pointer1 = len(valFeature)
            actLen = pointer1 - pointer0
            pointer0 = pointer1 - batch_size
            pointer0 = max(0, pointer0)
        else:
            pointer1 = pointer0 + batch_size
            actLen = batch_size
        batch_encoder_input = valFeature[pointer0:pointer1].reshape(
            [batch_size, n_steps_encoder, stationNumber])
        batchOtherFeature = valOtherFeature[pointer0:pointer1].reshape([batch_size, addFeatureLength])
        batch_preNN_input = valFeature[pointer0: pointer1][:, -1, :].reshape([batch_size, stationNumber])
        batch_preNN_target = valTarget[pointer0: pointer1].reshape([batch_size, stationNumber])
        finalState = autoEncoderSession.run(
            en_final_state,
            feed_dict={
                demandMaskTensorAutoEncoder: demandMaskTensorFeed,
                fusionGraphMatrixTensorRaw: fusionGraphMatrixFeed,
                encoderRawInput: batch_encoder_input,
            }
        )
        prediction = preNNSession.run(preOutput,
                              feed_dict={
                                  demandMaskTensorPreNN: demandMaskTensorFeed,
                                  # H
                                  inputEmbeddingFeature: finalState[-1],
                                  otherFeature: batchOtherFeature,
                                  # C, maybe c is better
                                  # inputEmbeddingFeature: finalState[0],
                                  preNNInputRaw: batch_preNN_input,
                                  preNNTargetRaw: batch_preNN_target
                              })
        if iteration == 0:
            tmpPreList = prediction[batch_size - actLen:batch_size]
        else:
            tmpPreList = np.concatenate((tmpPreList, prediction[batch_size - actLen:batch_size]), axis=0)
    valPredictionList.append(tmpPreList)
valPredictionList = np.array(valPredictionList, dtype=np.float32)

# compute the uncertainty
finalPreResult = np.mean(mulPredictionList, axis=0)
uncertainty = np.sqrt(np.var(mulPredictionList, axis=0) + np.mean(np.var(valPredictionList, 0), axis=0))

# save the result
np.savetxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-finalPreResult.txt'),
           np.array(finalPreResult, dtype=np.float32),
           newline='\n', delimiter=' ')
np.savetxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-uncertainty.txt'),
           np.array(uncertainty, dtype=np.float32),
           newline='\n', delimiter=' ')
np.savetxt(os.path.join(GraphDemandPreDataPath, codeVersion + '-testTarget.txt'),
           np.array(np.dot(testTarget, demandMaskTensorFeed).reshape([len(finalPreResult), -1]), dtype=np.float32),
           newline='\n', delimiter=' ')

autoEncoderSession.close()
preNNSession.close()