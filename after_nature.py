import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, Concatenate, Reshape, Activation, BatchNormalization, SeparableConv1D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from sklearn import preprocessing
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow as tf
import scipy.io as sio
from sklearn.model_selection import train_test_split


def label_encoding(labels):
	encoded_labels = np.empty((len(labels),5),dtype=int)
	for i in range(len(labels)):
		if labels[i] == 0:
			encoded_labels[i] = [1,0,0,0,0]
		elif labels[i] == 1:
			encoded_labels[i] = [0,1,0,0,0]
		elif labels[i] == 2:
			encoded_labels[i] = [0,0,1,0,0]
		elif labels[i] == 3:
			encoded_labels[i] = [0,0,0,1,0]
		elif labels[i] == 4:
			encoded_labels[i] = [0,0,0,0,1]
	return np.array(encoded_labels)
 
def inception_module_1(layer_in):
    conv1 = Conv1D(16, 1, padding='same', activation='relu', kernel_initializer='GlorotNormal',kernel_regularizer=l2(0.0002))(layer_in)
    conv4 = Conv1D(16, 4, padding='same', activation='relu', kernel_initializer='GlorotNormal',kernel_regularizer=l2(0.0002))(layer_in)
    conv16 = Conv1D(16, 16, padding='same', activation='relu', kernel_initializer='GlorotNormal',kernel_regularizer=l2(0.0002))(layer_in)
    layer_out = concatenate([conv1, conv4, conv16], axis=-1)
    return layer_out
    
def res_net_block1(input_data, filters, conv_size):
    x = Conv1D(filters, conv_size, activation='relu', padding='same', kernel_regularizer=l2(0.0002))(input_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters, conv_size, activation=None, padding='same', kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x

def res_net_block_trans(input_data, filters, conv_size):
    input_trans = Conv1D(filters, 1, activation='relu', padding='same', kernel_regularizer=l2(0.0002))(input_data)
    x0 = Conv1D(filters, conv_size, activation='relu', padding='same', kernel_regularizer=l2(0.0002))(input_data)
    #x1 = BatchNormalization()(x0)
    x2 = Conv1D(filters, conv_size, activation=None, padding='same', kernel_regularizer=l2(0.0002))(x0)
    #x3 = BatchNormalization()(x2)
    x4 = Add()([x2, input_trans])
    x = Activation('relu')(x4)
    return x

def define_model(in_shape=(268,1), out_shape=5, initial_bias=None):
    if initial_bias is not None:
      output_bias = tf.keras.initializers.Constant(initial_bias)
    input = Input(shape=(268,1))
    layer_out_0 = Conv1D(32, 8, activation='relu', padding='same',kernel_regularizer=l2(0.0002))(input)
    layer_out = inception_module_1(layer_out_0)
    Batch1 = BatchNormalization()(layer_out)
    Pool1 = AveragePooling1D(2, padding='same')(Batch1)
    res2 = res_net_block_trans(Pool1, 32, 4)
    Batch11 = BatchNormalization()(res2)
    Pool2 = AveragePooling1D(2, padding='same')(Batch11)
    layer_out2 = inception_module_1(Pool2)
    Batch3 = BatchNormalization()(layer_out2)
    Pool3 = AveragePooling1D(2, padding='same')(Batch3)
    Conv1_2 = res_net_block_trans(Pool3, 64, 4)
    Batch4 = BatchNormalization()(Conv1_2)
    Pool4 = AveragePooling1D(2, padding='same')(Batch4)
    flat_1 = Flatten()(Pool4)
    Dense_1 = Dense(64, activation='relu')(flat_1)
    Dropout2 = Dropout(rate=0.4)(Dense_1)
    out = Dense(out_shape, activation='softmax')(Dropout2)
    BerkenLeNet = Model(inputs=input, outputs=out)
    #BerkenLeNet.summary()
    # compile model
    opt = Adam(learning_rate=0.001)
    BerkenLeNet.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy','Recall', tfa.metrics.F1Score(num_classes=5, threshold=0.5, average='macro')])
    return BerkenLeNet

mat_contents = sio.loadmat('./CV1.mat')
CV = mat_contents['CV1']
val = CV[0, 0]
xtrain = val['xtrain']
ytrain = val['ytrain']
xval = val['xval']
yval = val['yval']
xtest = val['xtest']
ytest = val['ytest']
xtrain = xtrain[:, 3:271]
xval = xval[:, 3:271]
xtest = xtest[:, 3:271]
xtrain = np.concatenate((xtrain, xval), axis=0)
ytrain = np.concatenate((ytrain, yval), axis=0)


for ir in range(10):
    ir1 = np.random.randint(1,100)
    
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, stratify=ytrain, test_size=0.1, random_state=ir1)
    
    xtrain = np.reshape(xtrain, (len(xtrain), 268, 1))
    valX = np.reshape(xval, (len(xval), 268, 1))
    trainY1 = to_categorical(ytrain, num_classes=5, dtype='int')
    testY1 = to_categorical(ytest, num_classes=5, dtype='int')
    valY1 = to_categorical(yval, num_classes=5, dtype='int')
    
    W =  list(ytrain.flatten()).count(0)
    N1 = list(ytrain.flatten()).count(1)
    N2 = list(ytrain.flatten()).count(2)
    N3 = list(ytrain.flatten()).count(3)
    REM = list(ytrain.flatten()).count(4)
    
    initial_bias = np.log([W / len(ytrain), N1 / len(ytrain), N2 / len(ytrain), N3 / len(ytrain), REM / len(ytrain)])
    
    model = define_model(initial_bias=initial_bias)
    
    checkpoint_filepath = 'checkpoints'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=False,
        monitor='val_recall',
        mode='max',
        save_best_only=True)
            
    stop_me = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=100,
    verbose=1, mode='max', baseline=None, restore_best_weights=True)
    
    where_am_I = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_f1_score', factor=0.1, patience=75, verbose=1,
    mode='max', min_delta=0.001, cooldown=0, min_lr=0)
        
    history = model.fit(xtrain, trainY1, epochs=500, batch_size=250, verbose=0,
                        validation_data=(valX, valY1), callbacks=[stop_me, model_checkpoint_callback,where_am_I])
    
    hist_df = pd.DataFrame(history.history)
    pd.DataFrame.from_dict(history.history).to_csv('doubleIncep-250-Lr.csv', index=False)
    
    model= tf.keras.models.load_model(checkpoint_filepath, custom_objects={"F1Score": tfa.metrics.F1Score})

    xtrain = np.reshape(xtrain, (len(xtrain), 268, 1))
    xtest = np.reshape(xtest, (len(xtest), 268, 1))
    xval = np.reshape(xval, (len(xval), 268, 1))
    
    all_data = np.concatenate((xtrain, xtest, xval))
    all_labels = np.concatenate((ytrain, ytest, yval))
    
    y_pred = model.predict(all_data)
    
    prediction = np.argmax(y_pred, 1)
    
    print(confusion_matrix(all_labels, prediction))
    
    metric = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=True)
    metric1 = tfa.metrics.F1Score(num_classes=5)
    metric.update_state(all_labels , prediction)
    trainY = label_encoding(ytrain)
    metric1.update_state(np.concatenate((trainY,testY1,valY1)) , y_pred)
    result = metric.result()
    result1 = metric1.result()
    print('Kappa = ', result.numpy())
    print('F1 scores =', result1.numpy())
    print('CV= ', 6)
    print('ir1=', ir1)

    model.save('saved_model/my_model')
    
    print('exit')


