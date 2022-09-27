import io
import itertools
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from newSignalDataset import SignalsDataset
import tensorflow.keras.backend as K


class MaskRelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)


def get_loc_classifier(input_shapes,
                       args,
                       L,
                       loc_kernel_initializer = keras.initializers.he_uniform(),
                       class_kernel_initializer = keras.initializers.glorot_uniform()):


    inputLocSignalsShape = input_shapes[0]
    inputLocFeaturesShape = input_shapes[1]



    inputLocSignals = keras.Input(shape=inputLocSignalsShape)
    inputLocFeatures = keras.Input(shape=inputLocFeaturesShape)


    fusion = args.train_args['loc_fusion']
    mask = args.train_args['mask']
    nullLoc = args.train_args['nullLoc']
    padding_method = args.train_args['padding_method']

    if fusion in ['LSTM','BidirectionalLSTM','CNNLSTM','FCLSTM']:

        X = inputLocSignals
        Y = inputLocFeatures




        if nullLoc == 'masking':
            masking_layer = keras.layers.Masking(mask_value=mask, name='maskLayer1')
            X = masking_layer(X)

        if padding_method != 'variableLength':
            SBNLayer = keras.layers.BatchNormalization(name = 'SignalsLocBatch')
            X = SBNLayer(X)

        if fusion == 'LSTM':
            lstmLayer = keras.layers.LSTM(
                units = 128,
                name = 'locLSTM'
            )
            X = lstmLayer(X)

        elif fusion == 'BidirectionalLSTM':
            lstmLayer = Bidirectional(tf.keras.layers.LSTM(units=128),
                                      name = 'locBiLSTM')

            X = lstmLayer(X)

        elif fusion == 'CNNLSTM':
            cnnLayer = keras.layers.Conv2D(
                filters=1,
                kernel_size=(1,inputLocSignalsShape[1]),
                strides=1,
                padding='valid',
                name='locCNN',
                kernel_initializer=loc_kernel_initializer
            )


            lstmLayer = tf.keras.layers.LSTM(units = 256,
                                             name = 'locLSTM')

            X = cnnLayer(X)
            X = tf.squeeze(X, axis=-1)
            X = lstmLayer(X)


        elif fusion == 'FCLSTM':
            TDDenseLayer = TimeDistributed(
                keras.layers.Dense(
                    units = 32,
                    kernel_initializer = loc_kernel_initializer,
                    name = 'TDlocDense1'
                )
            )
            TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch1')
            TDactivationLayer = MaskRelu()

            X = TDDenseLayer(X)
            X = TDbnLayer(X)
            X = TDactivationLayer(X)

            TDDenseLayer = TimeDistributed(
                keras.layers.Dense(
                    units=32,
                    kernel_initializer=loc_kernel_initializer,
                    name='TDlocDense2'
                )
            )
            TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch2')
            TDactivationLayer = MaskRelu()

            X = TDDenseLayer(X)
            X = TDbnLayer(X)
            X = TDactivationLayer(X)




            lstmLayer1 = tf.keras.layers.LSTM(units = 64,
                                             return_sequences = True,
                                             name = 'locLSTM1')
            lstmLayer2 = tf.keras.layers.LSTM(units = 64,
                                             name = 'locLSTM2')



            X = lstmLayer1(X)
            X = lstmLayer2(X)



        dropoutLayer = keras.layers.Dropout(rate=args.train_args['input_dropout'])
        X = dropoutLayer(X)



        X = tf.concat([X,Y],axis=1)


        denseLayer = keras.layers.Dense(
            units = 128,
            kernel_initializer = loc_kernel_initializer,
            name = 'locDense1'
        )


        bnLayer = keras.layers.BatchNormalization(name = 'locBatch1')
        activationLayer = MaskRelu()
        dropoutLayer = keras.layers.Dropout(rate=0.25)

        X = denseLayer(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = dropoutLayer(X)

        denseLayer = keras.layers.Dense(
            units=64,
            kernel_initializer=loc_kernel_initializer,
            name = 'locDense2'
        )

        bnLayer = keras.layers.BatchNormalization(name = 'locBatch2')
        activationLayer = MaskRelu()
        dropoutLayer = keras.layers.Dropout(rate=0.25)

        X = denseLayer(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = dropoutLayer(X)

        denseLayer = keras.layers.Dense(
            units=L,
            kernel_initializer=loc_kernel_initializer,
            name = 'locDense3'
        )

        bnLayer = keras.layers.BatchNormalization(name='locBatch3')
        activationLayer = MaskRelu()
        dropoutLayer = keras.layers.Dropout(rate=0.25)

        X = denseLayer(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = dropoutLayer(X)

        mask_w = K.switch(
            tf.reduce_all(tf.equal(inputLocFeatures, mask), axis=1, keepdims=True),
            lambda: tf.zeros_like(X), lambda: tf.ones_like(X)
        )

        X = tf.multiply(X, mask_w)




    elif fusion == 'DNN':
        bnLayer = keras.layers.BatchNormalization()
        X = bnLayer(inputLoc)
        dropoutLayer = keras.layers.Dropout(rate=args.train_args['input_dropout'])
        X = dropoutLayer(X)


        denseLayer = keras.layers.Dense(
            units=64,
            kernel_initializer=loc_kernel_initializer
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
            kernel_initializer=loc_kernel_initializer
        )

        bnLayer = keras.layers.BatchNormalization()
        activationLayer = keras.layers.Activation('selu')
        dropoutLayer = keras.layers.Dropout(rate=0.25)

        X = denseLayer(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = dropoutLayer(X)


        denseLayer = keras.layers.Dense(
            units=L,
            kernel_initializer=loc_kernel_initializer
        )

        bnLayer = keras.layers.BatchNormalization()
        activationLayer = keras.layers.Activation('selu')
        dropoutLayer = keras.layers.Dropout(rate=0.25)

        X = denseLayer(X)
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

    return keras.models.Model(inputs = [inputLocSignals,inputLocFeatures],
                              outputs = y_pred,
                              name = 'LocationEncoder')



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
    def __init__(self, val, batchSize, steps, logdir):
        super(Metrics, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps
        self.file_writer = tf.summary.create_file_writer(logdir + '/scalars')

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.f1 = 0
        self.recall = 0
        self.precision = 0

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


        self.val_f1s.append(self.f1)
        self.val_recalls.append(self.recall)
        self.val_precisions.append(self.precision)

        logs['f1'] = self.f1
        logs['recall'] = self.recall
        logs['precision'] = self.precision

        with self.file_writer.as_default():
            with tf.name_scope('val_score'):
                tf.summary.scalar('f1_score', data = self.f1, step = epoch)
                tf.summary.scalar('recall', data = self.recall, step = epoch)
                tf.summary.scalar('precision', data = self.precision, step = epoch)

            tf.summary.scalar('learning_rate', data = self.model.optimizer.lr, step = epoch)


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

        print(" - test_f1: %f - test_precision: %f - test_recall %f" % (test_f1, test_precision, test_recall))

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

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" % (val_f1, val_precision, val_recall))

        cm = confusion_matrix(val_true,val_predict)
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
        verbose = 1):



    accBagSize = config_edit('train_args', 'accBagSize', 1)
    locBagSize = config_edit('train_args', 'locBagSize', 1)
    bagStride = config_edit('train_args', 'bagStride', 3)
    pair_threshold = config_edit('train_args', 'pair_threshold', 60000)
    percentage = config_edit('train_args', 'val_percentage', 0.4)

    SD = SignalsDataset()

    train , val , test = SD(baseline = True,
                            locTransfer = True)

    config_edit('train_args', 'accBagSize', accBagSize)
    config_edit('train_args', 'locBagSize', locBagSize)
    config_edit('train_args', 'bagStride', bagStride)
    config_edit('train_args', 'pair_threshold', pair_threshold)
    config_edit('train_args', 'val_percentage', percentage)

    # import copy
    # transformedAccBag = SD.accTfrm(copy.deepcopy(SD.acceleration[[0]]),
    #                                  is_train=True)
    #
    # return

    user = SD.shl_args.train_args['test_user']
    logdir = os.path.join('logs_user' + str(user), 'loc_transfer_tensorboard')

    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    val_steps = SD.valSize // SD.valBatchSize

    train_steps = SD.trainSize // SD.trainBatchSize

    test_steps = SD.testSize // SD.testBatchSize

    if SD.shl_args.train_args['loss_function'] == 'weighted':

        N = SD.trainSize

        lb_count = np.zeros(SD.n_labels)

        for index in SD.train_indices:
            lb_count = lb_count + SD.lbsTfrm(SD.labels[index, 0])

        Wp = tf.convert_to_tensor(N / (2. * lb_count), tf.float32)
        Wn = tf.convert_to_tensor(N / (2. * (N - lb_count)), tf.float32)

        loss_function = get_loss_function(Wp, Wn)

    elif SD.shl_args.train_args['loss_function'] == 'focal':
        loss_function = get_focal_loss()

    else:
        loss_function = keras.losses.CategoricalCrossentropy()

    Model = get_loc_classifier(SD.inputShape,
                               SD.shl_args,
                               L=L)


    Model.compile(
        optimizer = keras.optimizers.Adam(
            learning_rate=SD.shl_args.train_args['learning_rate']
        ),
        loss = loss_function,
        metrics = [keras.metrics.categorical_accuracy]
    )

    val_metrics = Metrics(val,
                         SD.valBatchSize,
                         val_steps,
                         logdir)

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


    model_type = 'location_classifier'
    model_name = 'shl_%s_model.h5' %model_type
    filepath = os.path.join(save_dir, model_name)

    save_model = keras.callbacks.ModelCheckpoint(
                filepath = filepath,
                monitor = 'val_loss',
                verbose = 1,
                save_best_only = True,
                mode = 'min'
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

    Model.fit(
        train,
        epochs=SD.shl_args.train_args['locEpochs'],
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = [tensorboard_callback,
                     val_metrics,val_cm,
                     save_model,
                     early_stopping,
                     reduce_lr_plateau],
        use_multiprocessing = True,
        verbose=verbose
    )

    Model.load_weights(filepath)

    if evaluation:


        test_cm = testConfusionMetric(test,
                                      SD.testBatchSize,
                                      test_steps,
                                      file_writer_test,
                                      drop_run = SD.shl_args.train_args['drop_run'])

        Model.evaluate(test,steps=test_steps,callbacks=[test_cm])



    return filepath