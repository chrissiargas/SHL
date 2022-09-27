import io
import itertools
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt





def getBaselineModel(input_shapes,args):
    inputAccShape = input_shapes[0]
    inputLocShape = input_shapes[1]

    inputAcc = keras.Input(shape=inputAccShape)
    inputLoc = keras.Input(shape=inputLocShape)

    useSpecto = args.train_args['spectograms']
    fusion = args.train_args['acc_fusion']

    X = inputAcc
    if not useSpecto:
        if fusion == 'Depth':
            length , channels = inputAccShape


            activationLayer = keras.layers.ReLU()

            poolingLayer = keras.layers.MaxPooling1D(2)

            for i in range(4,7):
                conv1D = keras.layers.Conv1D(
                    filters=channels*(2**i),
                    kernel_size=10,
                    strides=3,
                    padding='valid',
                    groups=1
                )

                X = conv1D(X)
                X = activationLayer(X)
                X = poolingLayer(X)

            flattenLayer = keras.layers.Flatten()
            X = flattenLayer(X)

            for i in range(7,4,-1):


                dropoutLayer = keras.layers.Dropout(rate=0.25)
                dnn = keras.layers.Dense(units=2**i)

                X = dropoutLayer(X)
                X = dnn(X)

    acc_encodings = X


    fusion = args.train_args['loc_fusion']

    X = inputLoc
    if fusion == 'DNN':


        for i in range(3):

            denseLayer = keras.layers.Dense(
                units=2**(7-i),
                kernel_initializer=keras.initializers.lecun_uniform()
            )
            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.Activation('selu')
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)

        denseLayer = keras.layers.Dense(
            units=32,
            kernel_initializer=keras.initializers.lecun_uniform()
        )
        bnLayer = keras.layers.BatchNormalization()
        activationLayer = keras.layers.Activation('selu')
        dropoutLayer = keras.layers.Dropout(rate=0.5)

        X = denseLayer(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = dropoutLayer(X)


    loc_encodings = X

    fusion_output = keras.layers.concatenate(
        [acc_encodings, loc_encodings]
    )

    X = fusion_output



    for i in range(4):
        if i==3:
            denseLayer = keras.layers.Dense(
                units=2 ** (7 - i),
                kernel_initializer=keras.initializers.lecun_uniform()
            )
            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.Activation('selu')
            dropoutLayer = keras.layers.Dropout(rate=0.5)

        else:
            denseLayer = keras.layers.Dense(
                units=2 ** (7 - i),
                kernel_initializer=keras.initializers.lecun_uniform()
            )
            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.Activation('selu')
            dropoutLayer = keras.layers.Dropout(rate=0.25)

        X = denseLayer(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = dropoutLayer(X)

    finalLayer = keras.layers.Dense(units=8,
                                    activation= 'softmax',
                                    kernel_initializer='he_uniform')

    output = finalLayer(X)

    model = keras.models.Model(inputs = [inputAcc,inputLoc], outputs = output)

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(),
                  metrics = [keras.metrics.categorical_accuracy])

    return model

class Metrics(keras.callbacks.Callback):
    def __init__(self, val, batchSize, steps):
        super(Metrics, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

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

        val_f1 = f1_score(val_true, val_predict, average="macro")
        val_recall = recall_score(val_true, val_predict, average="macro")
        val_precision = precision_score(val_true, val_predict, average="macro")

        del val_predict
        del val_true

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)

        print(" - val_f1: %f - val_precision: %f - val_recall %f" %(val_f1,val_precision,val_recall))


        return


class confusion_metric(keras.callbacks.Callback):
    def __init__(self,val, batchSize, steps, file_writer):
        super(confusion_metric, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps
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


        cm = confusion_matrix(val_true,val_predict)
        figure = plot_confusion_matrix(cm,class_names=self.class_names)
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
    plt.xticks(tick_marks, class_names, rotation=45)
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


def fit(SignalsDataset):

    train , test = SignalsDataset()

    Model = getBaselineModel(SignalsDataset.inputShape, SignalsDataset.shl_args)
    print(Model.summary())


    logdir = os.path.join(r'C:\Users\chris\PycharmProjects\shlProject\logs','tensoboard') #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    file_writer = tf.summary.create_file_writer(logdir + '/cm')


    val_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize
    train_steps = SignalsDataset.trainSize // SignalsDataset.trainBatchSize
    metrics = Metrics(test,SignalsDataset.testBatchSize,val_steps)
    cm = confusion_metric(test,SignalsDataset.testBatchSize,val_steps,file_writer)


    Model.fit(
        train,
        epochs = 10,
        steps_per_epoch = train_steps,
        validation_data = test,
        validation_steps = val_steps,
        callbacks = [tensorboard_callback, cm, metrics]
    )