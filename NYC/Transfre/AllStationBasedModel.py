from DataAPI.utils import *
import tensorflow as tf
from scipy import stats
from SharedParameters.SharedParameters import *

stationIDDict = getJsonData('centralStationIDList.json')
centralStationIDList = stationIDDict['centralStationIDList']
allStationIDList = stationIDDict['allStationIDList']

currentFileName = __file__.split('/')[-1][:-3]
codeVersion = currentFileName.replace('_', '-')

# (4) parameter for saving the model
AutoEncoderModelFileName = 'AutoEncoderModel-%s' % (codeVersion)
preModelFileName = 'PredModel-%s' % (codeVersion)
autoEncoderModelFileSavePath = os.path.join(GraphDemandPreDataPath, AutoEncoderModelFileName)
preModelFileSavePath = os.path.join(GraphDemandPreDataPath, preModelFileName)
if not os.path.exists(autoEncoderModelFileSavePath):
    os.makedirs(autoEncoderModelFileSavePath)
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

GraphValueData = getJsonData('GraphValueMatrix.json')

GraphValueMatrix = GraphValueData['GraphValueMatrix'] # day, 24, stationNumber
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

distanceGraphMatrix = np.loadtxt(os.path.join(txtPath, 'distanceGraphMatrix.txt'), delimiter=' ')
demandGraphMatrix = np.loadtxt(os.path.join(txtPath, 'demandGraphMatrix.txt'), delimiter=' ')
demandMask = np.loadtxt(os.path.join(txtPath, 'demandMask.txt'), delimiter=' ')

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

demandMaskTensorFeed = np.array(demandMask, dtype=np.float32)
distanceGraphMatrixFeed = np.array(distanceGraphMatrix, dtype=np.float32)

# 待预测站点的Index
targetStationIndexList = [(e, allStationIDList.index(centralStationIDList[e])) for e in range(len(centralStationIDList[:30]))]

###################################################################################################################
# Build The NetWork
#########################################################################
# Define Graphs
autoEncoderGraph = tf.Graph()
preNNGraph = tf.Graph()

with autoEncoderGraph.as_default():

    # Input
    encoderRawInput = tf.placeholder(tf.float32, [None, n_steps_encoder, stationNumber])
    decoderRawInput = tf.placeholder(tf.float32, [None, n_steps_decoder, stationNumber])
    decoderRawTarget = tf.placeholder(tf.float32, [None, len(targetStationIndexList)])

    # Encoder
    with tf.variable_scope('Encoder', reuse=False) as en:
        # Graph information
        distGraphMatrixTensorRaw = tf.placeholder(tf.float32, [stationNumber, stationNumber])
        distGraphMatrixWeight = tf.Variable(tf.random_normal([stationNumber, stationNumber]))
        # Graph Convolution
        encoderInput = tf.nn.tanh(tf.matmul(tf.reshape(encoderRawInput, [-1, stationNumber]),
                                            tf.multiply(distGraphMatrixTensorRaw, distGraphMatrixWeight), transpose_b=True))
        encoderInput = tf.reshape(encoderInput, [-1, n_steps_encoder, stationNumber])

        en_outputs_all = []
        en_final_state_all = []
        for index, LSTM_Cell_Index in targetStationIndexList:
            with tf.variable_scope('EncoderLSTM_%s' % index):
                # init lstm cell for each station
                encoderCell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, state_is_tuple=True)
                encoderCell = tf.nn.rnn_cell.DropoutWrapper(encoderCell, variational_recurrent=True,
                                                            dtype=tf.float32,
                                                            output_keep_prob=1 - dropout_pro)
                en_outputs, en_final_state = tf.nn.dynamic_rnn(encoderCell,
                                                               encoderInput[:, :, LSTM_Cell_Index:LSTM_Cell_Index+1],
                                                               dtype=tf.float32, time_major=False)
                en_outputs_all.append(en_outputs)
                en_final_state_all.append(en_final_state)

    # Decoder
    with tf.variable_scope('decoder') as vs:
        # All the station share the same lstm cell
        decoderCell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, state_is_tuple=True)
        decoderCell = tf.nn.rnn_cell.DropoutWrapper(decoderCell, variational_recurrent=True,
                                                    dtype=tf.float32,
                                                    output_keep_prob=1 - dropout_pro, )
        # FC layers
        decoderOutputWeight = tf.Variable(tf.random_normal([n_hidden_units, n_inputs]),
                                          name='decoderOutputWeight')
        decoderOutputBias = tf.Variable(tf.constant(0.1, shape=[n_inputs, ]), name='encoderInputBias')
        # result collector
        decoderOutputList = []
        # Loop for all the stations
        for index, LSTM_Cell_Index in targetStationIndexList:
            # get the corresponding decoder inputs for each station
            decoderInput = decoderRawInput[:, :, LSTM_Cell_Index:LSTM_Cell_Index+1]
            decoderState = en_final_state_all[index]

            de_outputs, _ = tf.nn.dynamic_rnn(decoderCell, decoderInput, initial_state=decoderState, time_major=False)
            de_outputs = tf.reshape(tf.matmul(de_outputs[:, -1, :], decoderOutputWeight)
                                    + decoderOutputBias, [-1, n_inputs])

            decoderOutputList.append(de_outputs)

        de_outputs_final = tf.concat(decoderOutputList, axis=-1)

    loss_autoEncoder = tf.reduce_mean(tf.square(de_outputs_final - decoderRawTarget))
    trainOperation_autoEncoder = tf.train.AdamOptimizer(lr).minimize(loss_autoEncoder)

    autoEncoderSaver = tf.train.Saver(max_to_keep=None)
    init_autoEncoder = tf.global_variables_initializer()

# prediction network
# fully connected
with preNNGraph.as_default():

    inputEmbeddingFeature = tf.placeholder(tf.float32, [None, len(targetStationIndexList), n_hidden_units])
    preNNTarget = tf.placeholder(tf.float32, [batch_size, len(targetStationIndexList)])
    otherFeature = tf.placeholder(tf.float32, [None, addFeatureLength])

    with tf.variable_scope('predict') as vs_p:
        W = [tf.Variable(tf.random_normal([n_units_pre[e], n_units_pre[e + 1]])) for e in range(n_units_pre.__len__() - 1)]
        B = [tf.Variable(tf.constant(0.1, shape=[n_units_pre[e], ])) for e in range(1, n_units_pre.__len__())]
        preOutputList = []
        for index, LSTM_Cell_Index in targetStationIndexList:
            preOutput = tf.concat([inputEmbeddingFeature[:, index, :], otherFeature], 1)
            # remove the last layer (for output - linearRegression)
            for i in range(n_units_pre.__len__() - 2):
                vs_p.reuse_variables()
                preOutput = tf.nn.softplus(tf.matmul(preOutput, W[i]) + B[i])
                preOutput = tf.nn.dropout(preOutput, keep_prob=1 - dropout_pro)
            preOutput = tf.matmul(preOutput, W[-1]) + B[-1]
            preOutputList.append(preOutput)

        tv = vs_p.trainable_variables()
        regularization_cost = 0.0001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

    preOutputFinal = tf.concat(preOutputList, axis=-1)
    loss_pre = tf.reduce_mean(tf.square(preNNTarget - preOutputFinal)) + regularization_cost

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
            batch_decoder_target = autoEncoderTarget[pointer0: pointer1].reshape([batch_size, stationNumber])[:, [e[1] for e in targetStationIndexList]]
            trainLoss, _ = autoEncoderSession.run([loss_autoEncoder, trainOperation_autoEncoder],
                                    feed_dict={
                                        distGraphMatrixTensorRaw: distanceGraphMatrixFeed,
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

#######################################################################################################################
preNNSession = tf.Session(config=config, graph=preNNGraph)

if trainPreModel:
    preNNSession.run(init_preNN)
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
            batch_preNN_target = preTarget[pointer0: pointer1].reshape([batch_size, stationNumber])[:, [e[1] for e in targetStationIndexList]]
            finalStateList = autoEncoderSession.run(
                en_final_state_all,
                feed_dict={
                    distGraphMatrixTensorRaw: distanceGraphMatrixFeed,
                    encoderRawInput: batch_encoder_input,
                }
            )

            lossPre, _, output = preNNSession.run([loss_pre, trainOperation_pre, preOutput],
                                          feed_dict={
                                              # H
                                              inputEmbeddingFeature:
                                                  np.concatenate([e[-1].reshape([-1, 1, n_hidden_units])
                                                                  for e in finalStateList], axis=1),
                                              otherFeature: batchOtherFeature,
                                              # C, maybe c is better
                                              preNNTarget: batch_preNN_target
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

#######################################################################################################################
# get the prediction and uncertainty
mulPredictionList = []
for varIter0 in range(varB):
    print(codeVersion, 'varIter0 : ', varIter0)
    finalStateList = autoEncoderSession.run(
        en_final_state_all,
        feed_dict={
            distGraphMatrixTensorRaw: distanceGraphMatrixFeed,
            encoderRawInput: testFeature,
        }
    )
    prediction = preNNSession.run(preOutput,
                                  feed_dict={
                                      # H
                                      inputEmbeddingFeature:
                                          np.concatenate([e[-1].reshape([-1, 1, n_hidden_units])
                                                          for e in finalStateList], axis=1),
                                      otherFeature: testOtherFeature,
                                  })
    mulPredictionList.append(prediction)

mulPredictionList = np.array(mulPredictionList, dtype=np.float32)

valPredictionList = []
for varIter1 in range(varB):
    print(codeVersion, 'varIter1 : ', varIter1)
    tmpPreList = []
    finalStateList = autoEncoderSession.run(
        en_final_state_all,
        feed_dict={
            distGraphMatrixTensorRaw: distanceGraphMatrixFeed,
            encoderRawInput: valFeature,
        }
    )
    prediction = preNNSession.run(preOutput,
                                  feed_dict={
                                      # H
                                      inputEmbeddingFeature:
                                          np.concatenate([e[-1].reshape([-1, 1, n_hidden_units])
                                                          for e in finalStateList], axis=1),
                                      otherFeature: valOtherFeature,
                                      # C, maybe c is better
                                      # inputEmbeddingFeature: finalState[0],
                                  })
    valPredictionList.append(prediction)

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
           np.array(testTarget, dtype=np.float32),
           newline='\n', delimiter=' ')

autoEncoderSession.close()
preNNSession.close()