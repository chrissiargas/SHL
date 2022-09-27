import io
import itertools
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Bidirectional
from newSignalDataset import SignalsDataset
import simCLR
import tensorflow_addons as tfa


def get_acc_classifier(input_shapes,
                       args,
                       L,
                       acc_kernel_initializer = keras.initializers.he_uniform(),
                       class_kernel_initializer = keras.initializers.glorot_uniform(),
                       simCLR_weights = None,
                       finetuning = True):

    inputAccShape = input_shapes

    inputAcc = keras.Input(shape = inputAccShape)

    useSpecto = args.train_args['spectograms']
    useFFT = args.train_args['FFT']
    fusion = args.train_args['acc_fusion']

    dimension = args.train_args['dimension']

    model = args.train_args['acc_model']

    X = inputAcc

    trainable = False if simCLR_weights and not finetuning else True
    simCLR =  True if simCLR_weights else False

    if useSpecto:
        if fusion in ['Depth','Frequency','Time']:
            _, _, channels = inputAccShape

            if model == 'ResNet':
                residuals = 4
                for res in range(1,residuals+1):
                    residual = 'Res' + str(res)
                    filters = channels * 2**(res+3)
                    bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch1',
                                                              trainable=trainable)
                    Y = bnLayer(X)


                    conv2D = keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer=acc_kernel_initializer,
                        name=residual + 'Conv1',
                        trainable=trainable
                    )

                    bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch2',
                                                              trainable=trainable)
                    activationLayer = keras.layers.ReLU()


                    Z = conv2D(Y)
                    Z = bnLayer(Z)
                    Z = activationLayer(Z)

                    shortcut1 = Z
                    conv2D = keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=acc_kernel_initializer,
                        name=residual + 'Conv2',
                        trainable=trainable
                    )

                    bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch3',
                                                              trainable=trainable)
                    activationLayer = keras.layers.ReLU()


                    Z = conv2D(Z)
                    Z = bnLayer(Z)
                    Z = keras.layers.add([shortcut1, Z])
                    Z = activationLayer(Z)

                    shortcut2 = Z
                    conv2D = keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=acc_kernel_initializer,
                        name=residual + 'Conv3',
                        trainable=trainable
                    )

                    bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch4',
                                                              trainable=trainable)


                    Z = conv2D(Z)
                    Z = bnLayer(Z)
                    Z = keras.layers.add([shortcut1, shortcut2, Z])

                    conv2D = keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=1,
                        strides=2,
                        padding='same',
                        kernel_initializer=acc_kernel_initializer,
                        name=residual + 'ConvIdentity',
                        trainable=trainable
                    )
                    Y = conv2D(Y)

                    X = keras.layers.Add()([Y,Z])

                    activationLayer = keras.layers.ReLU()


                    X = activationLayer(X)

                flattenLayer = keras.layers.Flatten()


                X = flattenLayer(X)


                dropoutLayer = keras.layers.Dropout(rate=0.25)
                dnn = keras.layers.Dense(units=128,
                                         kernel_initializer=acc_kernel_initializer,
                                         name='accDense1',
                                         trainable=trainable)
                activationLayer = keras.layers.ReLU()

                X = dropoutLayer(X)
                X = dnn(X)
                X = activationLayer(X)


                dropoutLayer = keras.layers.Dropout(rate=0.25)
                dnn = keras.layers.Dense(units=L,
                                         kernel_initializer=acc_kernel_initializer,
                                         name='accDense2',
                                         trainable=trainable)
                activationLayer = keras.layers.ReLU()


                X = dropoutLayer(X)
                X = dnn(X)
                X = activationLayer(X)

                dropoutLayer = keras.layers.Dropout(rate=0.25)
                X = dropoutLayer(X)

            else:

                bnLayer = keras.layers.BatchNormalization(name = 'accBatch1',
                                                          trainable= trainable)
                X = bnLayer(X)

                paddingLayer = keras.layers.ZeroPadding2D(padding=(1,1))
                conv2D = keras.layers.Conv2D(
                    filters=channels * 16,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    kernel_initializer = acc_kernel_initializer,
                    name = 'accConv1',
                    trainable=trainable
                )

                bnLayer = keras.layers.BatchNormalization(name = 'accBatch2',
                                                          trainable= trainable)

                activationLayer = keras.layers.ReLU()
                poolingLayer = keras.layers.MaxPooling2D((2,2),strides=2)

                X = paddingLayer(X)
                X = conv2D(X)
                X = bnLayer(X)
                X = activationLayer(X)
                X = poolingLayer(X)

                paddingLayer = keras.layers.ZeroPadding2D(padding=(1, 1))
                conv2D = keras.layers.Conv2D(
                    filters=channels * 32,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    kernel_initializer=acc_kernel_initializer,
                    name = 'accConv2',
                    trainable=trainable
                )

                bnLayer = keras.layers.BatchNormalization(name = 'accBatch3',
                                                          trainable= trainable)
                activationLayer = keras.layers.ReLU()
                poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

                X = paddingLayer(X)
                X = conv2D(X)
                X = bnLayer(X)
                X = activationLayer(X)
                X = poolingLayer(X)

                activationLayer = keras.layers.ReLU()
                poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)


                conv2D = keras.layers.Conv2D(
                    filters=channels * 64,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    kernel_initializer=acc_kernel_initializer,
                    name = 'accConv3',
                    trainable=trainable
                )
                bnLayer = keras.layers.BatchNormalization(name = 'accBatch4',
                                                          trainable= trainable)
                X = conv2D(X)
                X = bnLayer(X)
                X = activationLayer(X)
                X = poolingLayer(X)

                flattenLayer = keras.layers.Flatten()
                dropoutLayer = keras.layers.Dropout(rate = args.train_args['input_dropout'])

                X = flattenLayer(X)
                X = dropoutLayer(X)

                if simCLR:
                    dnn = keras.layers.Dense(units=dimension,
                                             kernel_initializer=acc_kernel_initializer,
                                             name = 'accDense1',
                                             trainable=trainable)

                else:
                    dnn = keras.layers.Dense(units=dimension,
                                             kernel_initializer=acc_kernel_initializer,
                                             name = 'accDense1',
                                             trainable=trainable)

                bnLayer = keras.layers.BatchNormalization(name = 'accBatch5',
                                                          trainable=trainable)

                activationLayer = keras.layers.ReLU()
                dropoutLayer = keras.layers.Dropout(rate=0.25)

                X = dnn(X)
                X = bnLayer(X)
                X = activationLayer(X)
                X = dropoutLayer(X)



                dnn = keras.layers.Dense(units=L,
                                         kernel_initializer=acc_kernel_initializer,
                                         name='accDense2')

                bnLayer = keras.layers.BatchNormalization(name='accBatch6')
                activationLayer = keras.layers.ReLU()
                dropoutLayer = keras.layers.Dropout(rate=0.25)

                X = dnn(X)
                X = bnLayer(X)
                X = activationLayer(X)
                X = dropoutLayer(X)


    elif useFFT:
        if fusion == 'Depth':
            _, channels = inputAccShape
            bnLayer = keras.layers.BatchNormalization(name='accBatch1',
                                                      trainable=trainable)
            X = bnLayer(X)

            paddingLayer = keras.layers.ZeroPadding1D(padding=1)
            conv1D = keras.layers.Conv1D(
                filters=channels * 16,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name='accConv1',
                trainable=trainable
            )

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

            X = paddingLayer(X)
            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            paddingLayer = keras.layers.ZeroPadding1D(padding=1)
            conv1D = keras.layers.Conv1D(
                filters=channels * 32,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name='accConv2',
                trainable=trainable
            )

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

            X = paddingLayer(X)
            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

            conv1D = keras.layers.Conv1D(
                filters=channels * 64,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name='accConv3',
                trainable=trainable
            )

            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            flattenLayer = keras.layers.Flatten()
            dropoutLayer = keras.layers.Dropout(rate=args.train_args['input_dropout'])

            X = flattenLayer(X)
            X = dropoutLayer(X)

            dnn = keras.layers.Dense(units=128,
                                     kernel_initializer=acc_kernel_initializer,
                                     name='accDense1',
                                     trainable=trainable)

            bnLayer = keras.layers.BatchNormalization(name='accBatch2',
                                                      trainable=trainable)

            activationLayer = keras.layers.ReLU()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = dnn(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)

            dnn = keras.layers.Dense(units=L,
                                     kernel_initializer=acc_kernel_initializer,
                                     name='accDense2')

            bnLayer = keras.layers.BatchNormalization(name='accBatch3')
            activationLayer = keras.layers.ReLU()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = dnn(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)




    else:
        if fusion == 'Depth':
            _, channels = inputAccShape

            bnLayer = keras.layers.BatchNormalization(name='accBatch1',
                                                      trainable=trainable)
            X = bnLayer(X)

            paddingLayer = keras.layers.ZeroPadding1D(padding=1)
            conv1D = keras.layers.Conv1D(
                filters=channels * 16,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name='accConv1',
                trainable=trainable
            )

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

            X = paddingLayer(X)
            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            paddingLayer = keras.layers.ZeroPadding1D(padding=1)
            conv1D = keras.layers.Conv1D(
                filters=channels * 32,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name='accConv2',
                trainable=trainable
            )

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

            X = paddingLayer(X)
            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

            conv1D = keras.layers.Conv1D(
                filters=channels * 64,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name='accConv3',
                trainable=trainable
            )

            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            flattenLayer = keras.layers.Flatten()
            dropoutLayer = keras.layers.Dropout(rate=args.train_args['input_dropout'])

            X = flattenLayer(X)
            X = dropoutLayer(X)

            dnn = keras.layers.Dense(units=128,
                                     kernel_initializer=acc_kernel_initializer,
                                     name='accDense1',
                                     trainable=trainable)

            bnLayer = keras.layers.BatchNormalization(name='accBatch2',
                                                      trainable=trainable)

            activationLayer = keras.layers.ReLU()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = dnn(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)

            dnn = keras.layers.Dense(units=L,
                                     kernel_initializer=acc_kernel_initializer,
                                     name='accDense2')

            bnLayer = keras.layers.BatchNormalization(name='accBatch3')
            activationLayer = keras.layers.ReLU()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = dnn(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)



    if args.train_args['drop_run']:
        finalLayer = keras.layers.Dense(units=7,
                                        activation='softmax',
                                        kernel_initializer=class_kernel_initializer)


    else:
        finalLayer = keras.layers.Dense(units=8,
                                        activation='softmax',
                                        kernel_initializer=class_kernel_initializer)

    y_pred = finalLayer(X)

    return keras.models.Model(inputs = inputAcc,
                              outputs = y_pred,
                              name = 'AccelerationEncoder')



class testMetrics(keras.callbacks.Callback):
    def __init__(self, test, batchSize, steps):
        super(testMetrics, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps

    def on_test_end(self, logs=None):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))

        for batch in self.test.take(self.steps):

            test_data = batch[0]
            test_target = batch[1]


            test_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(test_data)),axis=1)
            test_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(test_target,axis=1)
            step += 1

        test_f1 = f1_score(test_true, test_predict, average="macro")
        test_recall = recall_score(test_true, test_predict, average="macro")
        test_precision = precision_score(test_true, test_predict, average="macro")

        del test_predict
        del test_true

        print(" - test_f1: %f - test_precision: %f - test_recall %f" %(test_f1,test_precision,test_recall))

        return

class Metrics(keras.callbacks.Callback):
    def __init__(self, val, batchSize, steps):
        super(Metrics, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps


    def on_epoch_end(self, epoch, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))


        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            val_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(val_data)),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)
            step += 1

        self.f1 = f1_score(val_true, val_predict, average="macro")
        self.recall = recall_score(val_true, val_predict, average="macro")
        self.precision =precision_score(val_true, val_predict, average="macro")

        del val_predict
        del val_true

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(self.f1,self.precision,self.recall))


        return

class testConfusionMetric(keras.callbacks.Callback):
    def __init__(self, test, batchSize, steps, file_writer,drop_run):
        super(testConfusionMetric, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps

        if drop_run:
            self.class_names = [
                'Still',
                'Walking',
                'Bike',
                'Car',
                'Bus',
                'Train',
                'Subway'
            ]

        else:
            self.class_names = [
                'Still',
                'Walking',
                'Run',
                'Bike',
                'Car',
                'Bus',
                'Train',
                'Subway'
            ]


        self.file_writer = file_writer



    def on_test_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))

        for batch in self.test.take(self.steps):
            test_data = batch[0]
            test_target = batch[1]

            test_predict[step * self.batchSize: (step + 1) * self.batchSize] = \
                np.argmax(np.asarray(self.model.predict(test_data)), axis=1)
            test_true[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(test_target, axis=1)
            step += 1

        test_f1 = f1_score(test_true, test_predict, average="macro")
        test_recall = recall_score(test_true, test_predict, average="macro")
        test_precision = precision_score(test_true, test_predict, average="macro")


        print(" - test_f1: %f - test_precision: %f - test_recall %f" %(test_f1,test_precision,test_recall))

        cm = confusion_matrix(test_true, test_predict)
        cm_df = pd.DataFrame(cm,
                             index=self.class_names,
                             columns=self.class_names)
        cm_df = cm_df.astype('float') / cm.sum(axis=1)[:,np.newaxis]

        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        cm_image = plot_to_image(figure)

        with self.file_writer.as_default():
            tf.summary.image('Confusion Matrix', cm_image, step=1)

        return


class confusion_metric(keras.callbacks.Callback):
    def __init__(self,val, batchSize, steps, file_writer,drop_run):
        super(confusion_metric, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps

        if drop_run:
            self.class_names = [
                'Still',
                'Walking',
                'Bike',
                'Car',
                'Bus',
                'Train',
                'Subway'
            ]

        else:
            self.class_names = [
                'Still',
                'Walking',
                'Run',
                'Bike',
                'Car',
                'Bus',
                'Train',
                'Subway'
            ]


        self.file_writer = file_writer

    def on_train_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))

        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            val_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(val_data)),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)
            step += 1

        val_f1 = f1_score(val_true, val_predict, average="macro")
        val_recall = recall_score(val_true, val_predict, average="macro")
        val_precision =precision_score(val_true, val_predict, average="macro")

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(val_f1,val_precision,val_recall))

        cm = confusion_matrix(val_true,val_predict)
        global CM
        CM = cm / cm.sum(axis=1)[:, np.newaxis]

        cm_df = pd.DataFrame(cm,
                             index = self.class_names,
                             columns = self.class_names)
        cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        figure = plt.figure(figsize=(10,10))
        sns.heatmap(cm_df, annot = True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        cm_image = plot_to_image(figure)

        with self.file_writer.as_default():
            tf.summary.image('Confusion Matrix', cm_image, step=1)

        return


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:,np.newaxis], decimals=3)
    threshold = cm.max() / 4.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        color = "white" if cm[i,j] > threshold else 'black'
        plt.text(j,i,cm[i,j],horizontalalignment='center',color=color)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image,0)

    return image

import tensorflow.keras.backend as K
def get_loss_function(Wp,Wn):

    def weighted_categorical_crossentropy(y_true,y_pred):


        first_term = Wp * y_true * tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0))
        second_term = Wn * (1. - y_true) * tf.math.log(tf.clip_by_value(1.-y_pred,1e-10,1.))

        loss = -tf.reduce_sum(first_term + second_term, axis=-1)

        return loss

    return weighted_categorical_crossentropy



def get_focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

import ruamel.yaml

def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            original_value = data[args][param]
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)

    return original_value


def fit(evaluation = True,
        L = 64,
        summary = True,
        verbose = 1,
        use_simCLR = 'none',
        postprocess = False,
        round = True):

    if use_simCLR == 'train':

        saved_model = simCLR.fit(
            evaluation=evaluation,
            summary=summary,
            verbose=verbose
        )

    elif use_simCLR == 'load':

        save_dir = os.path.join('training', 'saved_models')
        if not os.path.isdir(save_dir):
            return

        model_type = 'simCLR'
        model_name = 'shl_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)
        saved_model = filepath

    else:
        saved_model = None

    accBagSize = config_edit('train_args', 'accBagSize', 1)
    locBagSize = config_edit('train_args', 'locBagSize', 1)
    bagStride = config_edit('train_args', 'bagStride', 1)

    print(accBagSize,locBagSize,bagStride)

    SD = SignalsDataset()

    train , val , test = SD(baseline = True,
                            accTransfer = True,
                            postprocess=postprocess,
                            round=round)

    config_edit('train_args', 'accBagSize', accBagSize)
    config_edit('train_args', 'locBagSize', locBagSize)
    config_edit('train_args', 'bagStride', bagStride)

    # import copy
    # transformedAccBag = SD.accTfrm(copy.deepcopy(SD.acceleration[[0]]),
    #                                  is_train=True)
    #
    # return

    user = SD.shl_args.train_args['test_user']
    logdir = os.path.join('logs_user' + str(user),'transfer_tensorboard')

    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    val_steps = SD.valSize // SD.valBatchSize

    train_steps = SD.trainSize // SD.trainBatchSize

    test_steps = (SD.testSize // SD.testBatchSize)

    if SD.shl_args.train_args['loss_function'] == 'weighted':

        N = SD.trainSize

        lb_count = np.zeros(SD.n_labels)

        for index in SD.sensor_train_indices:
            lb_count = lb_count + SD.lbsTfrm(SD.labels[index, 0])

        Wp = tf.convert_to_tensor(N / (2. * lb_count), tf.float32)
        Wn = tf.convert_to_tensor(N / (2. * ( N - lb_count )), tf.float32)

        loss_function = get_loss_function(Wp, Wn)

    elif SD.shl_args.train_args['loss_function'] == 'focal':
        loss_function = get_focal_loss()

    else:
        loss_function = keras.losses.CategoricalCrossentropy()

    lr = SD.shl_args.train_args['learning_rate']
    finetuning = SD.shl_args.train_args['simCLR_finetuning']
    fn_lr_factor = SD.shl_args.train_args['finetuning_lr_factor']



    Model = get_acc_classifier(SD.inputShape,
                               SD.shl_args,
                               simCLR_weights = saved_model,
                               L=L, finetuning=finetuning)

    if saved_model:
        Model.load_weights(saved_model, by_name=True)


    if saved_model and finetuning:
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate= lr * fn_lr_factor),
            tf.keras.optimizers.Adam(learning_rate= lr)
        ]
        optimizer_per_layer = [(optimizers[0], Model.layers[:-5]), (optimizers[1], Model.layers[-5:])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizer_per_layer)

    else:
        optimizer = keras.optimizers.Adam(
            learning_rate= lr
        )



    Model.compile(
        optimizer = optimizer,
        loss = loss_function,
        metrics = [keras.metrics.categorical_accuracy]
    )

    val_metrics = Metrics(val,
                         SD.valBatchSize,
                         val_steps)

    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')

    val_cm = confusion_metric(val,
                              SD.valBatchSize,
                              val_steps,
                              file_writer_val,
                              drop_run = SD.shl_args.train_args['drop_run'])

    save_dir = os.path.join('training','saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    model_type = 'acceleration_classifier'
    model_name = 'shl_%s_model.h5' %model_type
    filepath = os.path.join(save_dir, model_name)

    save_model = keras.callbacks.ModelCheckpoint(
                filepath = filepath,
                monitor = 'val_loss',
                verbose = 1,
                save_best_only = True,
                mode = 'min',
                save_weights_only=True
        )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 30,
        mode = 'min',
        verbose = 1
    )

    reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.4,
        patience = 10,
        verbose = 1,
        mode = 'min'
    )

    save_model._supports_tf_logs = False

    if summary:
        print(Model.summary())

    if saved_model and finetuning:
        callbacks = [tensorboard_callback,
                     val_metrics,val_cm,
                     save_model,
                     early_stopping]

    else:
        callbacks = [tensorboard_callback,
                     val_metrics,val_cm,
                     save_model,
                     early_stopping,
                     reduce_lr_plateau]

    Model.fit(
        train,
        epochs=SD.shl_args.train_args['accEpochs'],
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = callbacks,
        use_multiprocessing = True,
        verbose=verbose
    )

    if evaluation:


        test_cm = testConfusionMetric(test,
                                      SD.testBatchSize,
                                      test_steps,
                                      file_writer_test,
                                      drop_run = SD.shl_args.train_args['drop_run'])

        Model.evaluate(test,steps=test_steps,callbacks=[test_cm])



    return filepath