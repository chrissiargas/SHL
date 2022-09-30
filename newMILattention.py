import io
import itertools
import shutil

import matplotlib
import sklearn.metrics

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import tree
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Bidirectional, TimeDistributed
import transferLearning

import locTransferLearning
import tensorflow.keras.backend as K

from hmmlearn import hmm
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense

class postprocessModel(keras.Model):
    def __init__(self,
                 input_shapes,
                 loss_function = tf.keras.losses.CategoricalCrossentropy(),
                 L = 16, D = 32,
                 gated = True):
        super(postprocessModel, self).__init__()
        self.L = L
        self.D = D
        self.K = 1
        self.gated = gated

        self.batchSize = 4
        self.input_shapes = input_shapes

        self.forward_encoder = self.get_for_encoder(

        )

        self.backward_encoder = self.get_back_encoder(

        )

        self.attention_layer = self.get_attention_layer(

        )

        self.classifier = self.get_classifier(

        )

        self.loss_estimator = loss_function

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.forward_encoder.summary()
        self.backward_encoder.summary()
        self.attention_layer.summary()
        self.classifier.summary()

    def compile(self, optimizer, **kwargs):
        super(postprocessModel, self).compile(**kwargs)

        self.optimizer = optimizer
        self.loss_tracker = keras.metrics.Mean(name = 'Loss')
        self.accuracy_tracker = keras.metrics.CategoricalAccuracy(name = 'Accuracy')

    def get_for_encoder(self, dropout = True):
        dropout_ = 0.25 if dropout else 0.

        input_shape = list(self.input_shapes[0])
        input_X = keras.Input(shape=(*input_shape, ))

        X = input_X
        rec_layer = LSTM(16, return_sequences=True, dropout=dropout_)
        X = rec_layer(X)

        rec_layer = LSTM(32, return_sequences=True, dropout=dropout_)
        X = rec_layer(X)

        rec_layer = LSTM(16, return_sequences=True, dropout=dropout_)
        X = rec_layer(X)

        rec_layer = LSTM(self.L)
        output_X = rec_layer(X)

        return keras.models.Model(
            inputs = input_X,
            outputs = output_X,
            name = 'ForwardEncoder'
        )

    def get_back_encoder(self, dropout = True):
        dropout_ = 0.25 if dropout else 0.

        input_shape = list(self.input_shapes[1])
        input_X = keras.Input(shape=(*input_shape,))

        X = input_X
        rec_layer = LSTM(16, return_sequences=True, dropout=dropout_)
        X = rec_layer(X)

        rec_layer = LSTM(32, return_sequences=True, dropout=dropout_)
        X = rec_layer(X)

        rec_layer = LSTM(16, return_sequences=True, dropout=dropout_)
        X = rec_layer(X)

        rec_layer = LSTM(self.L)
        output_X = rec_layer(X)

        return keras.models.Model(
            inputs=input_X,
            outputs=output_X,
            name='BackwardEncoder'
        )

    def get_attention_layer(self,
                            kernel_initializer = keras.initializers.glorot_normal(),
                            kernel_regularizer = keras.regularizers.l2(0.01)):


        encodings = keras.Input(shape = (self.L))
        D_layer = keras.layers.Dense(units=self.D,
                                     activation='tanh',
                                     kernel_initializer = kernel_initializer,
                                     kernel_regularizer = kernel_regularizer)

        if self.gated:
            G_layer = keras.layers.Dense(units=self.D,
                                         activation='sigmoid',
                                         kernel_initializer = kernel_initializer,
                                         kernel_regularizer = kernel_regularizer)

        K_layer = keras.layers.Dense(units=self.K,
                                     kernel_initializer = kernel_initializer)

        attention_weights = D_layer(encodings)

        if self.gated:
            attention_weights = attention_weights * G_layer(encodings)

        attention_weights = K_layer(attention_weights)

        return keras.models.Model(inputs = encodings ,
                              outputs = attention_weights,
                              name = 'AttentionLayer')


    def get_classifier(self,
                       kernel_initializer = keras.initializers.glorot_normal()):


        pooling = keras.Input(shape=(self.L))

        X = pooling

        finalLayer = keras.layers.Dense(units=8,
                                        activation='softmax',
                                        kernel_initializer=kernel_initializer)


        y_pred = finalLayer(X)


        return keras.models.Model(inputs = pooling,
                                  outputs = y_pred,
                                  name = 'Classifier')

    def forward_pass(self, for_inputs, back_inputs, batchSize = None):
        batchSize = batchSize if batchSize else self.batchSize



        forEncodings = self.forward_encoder(for_inputs)
        backEncodings = self.backward_encoder(back_inputs)

        forEncodings = tf.reshape(forEncodings, [batchSize, 1, self.L])
        backEncodings = tf.reshape(backEncodings, [batchSize, 1, self.L])

        Encodings = tf.keras.layers.concatenate([forEncodings, backEncodings], axis=-2)
        Encodings = tf.reshape(Encodings, (batchSize*2,self.L))

        attention_weights = self.attention_layer(Encodings)
        attention_weights = tf.reshape(attention_weights,[batchSize,2])
        softmax = keras.layers.Softmax()
        attention_weights = tf.expand_dims(softmax(attention_weights), -2)

        Encodings = tf.reshape(Encodings, [batchSize, 2, self.L])

        if self.batchSize == 1:
            pooling = tf.expand_dims(tf.squeeze(tf.matmul(attention_weights, Encodings)),axis= 0)

        else:
            pooling = tf.squeeze(tf.matmul(attention_weights, Encodings))



        y_pred = self.classifier(pooling)


        return y_pred

    def train_step(self, data):
        x, y = data
        forward_x, backward_x = x

        with tf.GradientTape() as tape:
            y_ = self.forward_pass(forward_x, backward_x)

            loss = self.loss_estimator(y, y_)

        learnable_weights = self.classifier.trainable_weights + \
                            self.attention_layer.trainable_weights + \
                            self.forward_encoder.trainable_weights + \
                            self.backward_encoder.trainable_weights

        grads = tape.gradient(
            loss,
            learnable_weights
        )

        self.optimizer.apply_gradients(
            zip(
                grads,
                learnable_weights
            )
        )

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y, y_)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        forward_x, backward_x = x

        y_ = self.forward_pass(forward_x, backward_x)

        loss = self.loss_estimator(y, y_)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y, y_)

        return {m.name: m.result() for m in self.metrics}

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):

        batchSize = inputs[0].shape[0]

        pred = self.forward_pass(
            inputs[0],
            inputs[1],
            batchSize
        )

        return pred





    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_tracker
        ]


class MaskRelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)


class valMetrics(keras.callbacks.Callback):
    def __init__(self, val, batchSize, steps, score = 'macro'):
        super(valMetrics, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps
        self.score = score

    def on_epoch_end(self, epoch, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))

        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            val_predict[step * self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(val_data)),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)
            step += 1

        f1 = f1_score(val_true, val_predict, average=self.score)
        recall = recall_score(val_true, val_predict, average=self.score)
        precision =precision_score(val_true, val_predict, average=self.score)

        del val_predict
        del val_true

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(f1,precision,recall))

        return

class testMetrics(keras.callbacks.Callback):
    def __init__(self, test, batchSize, steps, score = 'macro'):
        super(testMetrics, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps
        self.score = score

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

        test_f1 = f1_score(test_true, test_predict, average=self.score)
        test_recall = recall_score(test_true, test_predict, average=self.score)
        test_precision = precision_score(test_true, test_predict, average=self.score)

        del test_predict
        del test_true

        print(" - test_f1: %f - test_precision: %f - test_recall %f" %(test_f1,test_precision,test_recall))

        return

class confusion_metric(keras.callbacks.Callback):
    def __init__(self,
                 val,
                 batchSize,
                 accBagSize,
                 locBagSize,
                 steps,
                 file_writer,
                 weights_file_writer,
                 drop_run):
        super(confusion_metric, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps
        self.accBagSize = accBagSize
        self.locBagSize = locBagSize

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
        self.weights_file_writer = weights_file_writer


    def on_train_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))
        weights = np.zeros((total,2))
        nulls = np.zeros((total))

        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            w, pred = self.model.call(val_data, return_weights=True)
            val_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(pred),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)


            nulls[step * self.batchSize: (step+1) * self.batchSize] = np.array(
                [np.sum(np.count_nonzero(loc == -10000000, axis=2) != 0) for loc in val_data[1]]
            )

            weights[step * self.batchSize: (step+1) * self.batchSize] = np.concatenate([np.sum(w[:,:self.accBagSize],axis=1)[:,np.newaxis],
                                                                        np.sum(w[:,self.accBagSize:],axis=1)[:,np.newaxis]], axis=1)

            step += 1

        val_f1 = f1_score(val_true, val_predict, average="macro")
        val_recall = recall_score(val_true, val_predict, average="macro")
        val_precision =precision_score(val_true, val_predict, average="macro")

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(val_f1,val_precision,val_recall))


        cm = confusion_matrix(val_true,val_predict)
        global CM
        CM = cm/ cm.sum(axis=1)[:, np.newaxis]

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

        wm_pred = np.concatenate([val_predict[:, np.newaxis], nulls[:, np.newaxis], weights], axis=1)
        wm_pred_df = pd.DataFrame(
            wm_pred,
            columns=['class', 'nulls', 'accWeight', 'locWeight']
        )

        wm_pred_nulls = wm_pred_df.groupby(['class', 'nulls'], as_index=False).mean()
        wm_pred_nulls = pd.pivot_table(wm_pred_nulls, values="accWeight", index=["class"], columns=["nulls"],
                                       fill_value=0)
        wm_pred_null_count = wm_pred_df.groupby(['class', 'nulls']).size().to_frame(name='size').reset_index()
        wm_pred_null_count = pd.pivot_table(wm_pred_null_count, values="size", index=["class"], columns=["nulls"],
                                            fill_value=0)
        wm_pred_null_count = wm_pred_null_count.astype('float') / wm_pred_null_count.sum(axis=1)[:, np.newaxis]

        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12,16))
        fig.subplots_adjust(wspace=0.01)
        sns.heatmap(wm_pred_nulls, ax=ax, cbar=False, annot=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
        sns.heatmap(wm_pred_null_count, ax=ax2, cbar=False, annot=True)
        fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
        fig.suptitle('Weight Matrix')
        plt.ylabel('')
        plt.xlabel('nulls')
        ax2.yaxis.tick_right()
        ax.set_yticklabels(labels=self.class_names,
                           rotation=45)

        ax2.tick_params(rotation=0)

        wm_image = plot_to_image(fig)

        with self.weights_file_writer.as_default():
            tf.summary.image('Weight Matrix', wm_image, step=1)

        return


class testConfusionMetric(keras.callbacks.Callback):
    def __init__(self, test, batchSize, accBagSize, locBagSize, steps, file_writer, weights_file_writer, drop_run):
        super(testConfusionMetric, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps
        self.accBagSize = accBagSize
        self.locBagSize = locBagSize

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
        self.weights_file_writer = weights_file_writer

    def on_test_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))
        weights = np.zeros((total,2))
        nulls = np.zeros((total))


        for batch in self.test.take(self.steps):
            test_data = batch[0]
            test_target = batch[1]
            w, pred = self.model.call(test_data, return_weights = True)
            test_predict[step * self.batchSize: (step + 1) * self.batchSize] = \
                np.argmax(np.asarray(pred), axis=1)
            test_true[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(test_target, axis=1)


            nulls[step * self.batchSize: (step+1) * self.batchSize] = np.array(
                [np.sum(np.count_nonzero(loc == -10000000, axis=2) != 0) for loc in test_data[1]]
            )
            weights[step * self.batchSize: (step+1) * self.batchSize] = np.concatenate([np.sum(w[:,:self.accBagSize],axis=1)[:,np.newaxis],
                                                                        np.sum(w[:,self.accBagSize:],axis=1)[:,np.newaxis]], axis=1)
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

        wm_pred = np.concatenate([test_predict[:, np.newaxis], nulls[:, np.newaxis], weights], axis=1)
        wm_pred_df = pd.DataFrame(
            wm_pred,
            columns=['class', 'nulls', 'accWeight', 'locWeight']
        )

        wm_pred_nulls = wm_pred_df.groupby(['class', 'nulls'], as_index=False).mean()
        wm_pred_nulls = pd.pivot_table(wm_pred_nulls, values="accWeight", index=["class"], columns=["nulls"],
                                       fill_value=0)
        wm_pred_null_count = wm_pred_df.groupby(['class', 'nulls']).size().to_frame(name='size').reset_index()
        wm_pred_null_count = pd.pivot_table(wm_pred_null_count, values="size", index=["class"], columns=["nulls"],
                                            fill_value=0)
        wm_pred_null_count = wm_pred_null_count.astype('float') / wm_pred_null_count.sum(axis=1)[:, np.newaxis]

        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12,16))
        fig.subplots_adjust(wspace=0.01)
        sns.heatmap(wm_pred_nulls, ax=ax, cbar=False, annot=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
        sns.heatmap(wm_pred_null_count, ax=ax2, cbar=False, annot=True)
        fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
        fig.suptitle('Weight Matrix')
        plt.ylabel('')
        plt.xlabel('nulls')
        ax2.yaxis.tick_right()
        ax.set_yticklabels(labels=self.class_names,
                           rotation=45)

        ax2.tick_params(rotation=0)

        wm_image = plot_to_image(fig)

        with self.weights_file_writer.as_default():
            tf.summary.image('Weight Matrix', wm_image, step=1)

        return




def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image,0)

    return image

class newMILattention(keras.Model):
    def __init__(self,
                 input_shapes,
                 args,
                 loss_function = tf.keras.losses.CategoricalCrossentropy,
                 L = 32, D = 128,
                 acc_weights = None,
                 loc_weights = None,
                 fusion = 'MIL'):

        super(newMILattention, self).__init__()
        self.L = L
        self.D = D
        self.K = 1
        self.batchSize = args.train_args['trainBatchSize']
        self.valBatchSize = args.train_args['valBatchSize']
        self.fusion = fusion


        self.input_shapes = input_shapes
        self.args = args
        self.gated = self.args.train_args['use_gated']
        self.seperateMIL = self.args.train_args['seperate_MIL']

        self.mask = self.args.train_args['mask']

        transfer_learning_acc = True if acc_weights else False

        self.acc_encoder = self.get_acc_encoder(
            kernel_initializer=keras.initializers.he_uniform(),
            transfer = transfer_learning_acc, dimension=args.train_args['dimension']
        )

        if transfer_learning_acc:

            self.acc_encoder.load_weights(acc_weights,by_name=True)

            self.acc_encoder.trainable = False
            self.transferLearningAcc = True

        else:
            self.transferLearningAcc = False



        transfer_learning_loc = True if loc_weights else False

        self.loc_encoder = self.get_loc_encoder(
            kernel_initializer=keras.initializers.he_uniform(),
            transfer=transfer_learning_loc
        )

        if transfer_learning_loc:

            self.loc_encoder.load_weights(loc_weights, by_name=True)

            self.loc_encoder.trainable = False
            self.transferLearningLoc = True


        else:
            self.transferLearningLoc = False


        if self.fusion == 'MIL':
            if not self.seperateMIL:
                self.attention_layer = self.get_attention_layer(
                    kernel_initializer=keras.initializers.glorot_uniform()
                )

            else:
                self.acc_attention_layer = self.get_attention_layer(data='acceleration')
                self.loc_attention_layer = self.get_attention_layer(data='location')

        self.classifier = self.get_classifier(
            kernel_initializer=keras.initializers.glorot_uniform()
        )

        self.loss_estimator = loss_function

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.acc_encoder.summary()
        self.loc_encoder.summary()

        if self.fusion == 'MIL':
            if not self.seperateMIL:
                self.attention_layer.summary()

            else:
                self.acc_attention_layer.summary()
                self.loc_attention_layer.summary()

        self.classifier.summary()

    def compile(self, optimizer, **kwargs):
        super(newMILattention, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_tracker = keras.metrics.Mean(name = 'Loss')
        self.accuracy_tracker = keras.metrics.CategoricalAccuracy(name = 'Accuracy')




    def get_acc_encoder(self,
                        kernel_initializer = keras.initializers.he_uniform(),
                        transfer = False, dropout = False, dimension = 128):

        inputAccShape = list(self.input_shapes[0])[1:]

        inputAcc = keras.Input( shape = (*inputAccShape, ) )

        useSpecto = self.args.train_args['spectograms']
        useFFT = self.args.train_args['FFT']
        fusion = self.args.train_args['acc_fusion']

        model = self.args.train_args['acc_model']

        X = inputAcc


        if useSpecto:
            if fusion in ['Depth', 'Frequency', 'Time']:

                _, _, channels = inputAccShape

                if model == 'ResNet':
                    residuals = 4
                    for res in range(1, residuals + 1):
                        residual = 'Res' + str(res)
                        bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch1')
                        Y = bnLayer(X)

                        conv2D = keras.layers.Conv2D(
                            filters=channels * 32,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_initializer=kernel_initializer,
                            name=residual + 'Conv1'
                        )

                        bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch2')
                        activationLayer = keras.layers.ReLU()

                        Z = conv2D(Y)
                        Z = bnLayer(Z)
                        Z = activationLayer(Z)

                        conv2D = keras.layers.Conv2D(
                            filters=channels * 32,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_initializer=kernel_initializer,
                            name=residual + 'Conv2'
                        )

                        bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch3')
                        activationLayer = keras.layers.ReLU()

                        Z = conv2D(Z)
                        Z = bnLayer(Z)
                        Z = activationLayer(Z)

                        conv2D = keras.layers.Conv2D(
                            filters=channels * 32,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_initializer=kernel_initializer,
                            name=residual + 'Conv3'
                        )

                        bnLayer = keras.layers.BatchNormalization(name=residual + 'Batch4')

                        Z = conv2D(Z)
                        Z = bnLayer(Z)

                        X = keras.layers.Add()([Y, Z])

                        activationLayer = keras.layers.ReLU()
                        poolingLayer = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)

                        X = activationLayer(X)
                        X = poolingLayer(X)

                    flattenLayer = keras.layers.Flatten()

                    X = flattenLayer(X)

                    dropoutLayer = keras.layers.Dropout(rate=0.25)
                    dnn = keras.layers.Dense(units=256,
                                             kernel_initializer=kernel_initializer,
                                             name='accDense1')
                    activationLayer = keras.layers.ReLU()

                    if not transfer or dropout:
                        X = dropoutLayer(X)
                    X = dnn(X)
                    X = activationLayer(X)

                    dropoutLayer = keras.layers.Dropout(rate=0.25)
                    dnn = keras.layers.Dense(units=self.L,
                                             kernel_initializer=kernel_initializer,
                                             name='accDense2')
                    activationLayer = keras.layers.ReLU()

                    if not transfer or dropout:
                        X = dropoutLayer(X)
                    X = dnn(X)
                    X = activationLayer(X)

                    dropoutLayer = keras.layers.Dropout(rate=0.25)
                    if not transfer or dropout:
                        X = dropoutLayer(X)

                else:
                    bnLayer = keras.layers.BatchNormalization(name = 'accBatch1')
                    X = bnLayer(X)

                    paddingLayer = keras.layers.ZeroPadding2D(padding=(1, 1))
                    conv2D = keras.layers.Conv2D(
                        filters=channels * 16,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        name='accConv1',
                        kernel_initializer = kernel_initializer
                    )
                    bnLayer = keras.layers.BatchNormalization(name = 'accBatch2')

                    activationLayer = keras.layers.ReLU()
                    poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

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
                        name='accConv2',
                        kernel_initializer=kernel_initializer
                    )
                    bnLayer = keras.layers.BatchNormalization(name = 'accBatch3')
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
                        name='accConv3',
                        kernel_initializer=kernel_initializer
                    )
                    bnLayer = keras.layers.BatchNormalization(name = 'accBatch4')

                    X = conv2D(X)
                    X = bnLayer(X)
                    X = activationLayer(X)
                    X = poolingLayer(X)

                    flattenLayer = keras.layers.Flatten()
                    dropoutLayer = keras.layers.Dropout(rate=self.args.train_args['input_dropout'])

                    X = flattenLayer(X)
                    if not transfer or dropout:
                        X = dropoutLayer(X)

                    dnn = keras.layers.Dense(units=dimension,
                                             name = 'accDense1',
                                             kernel_initializer = kernel_initializer)
                    bnLayer = keras.layers.BatchNormalization(name = 'accBatch5')
                    activationLayer = keras.layers.ReLU()
                    dropoutLayer = keras.layers.Dropout(rate=0.25)

                    X = dnn(X)
                    X = bnLayer(X)
                    X = activationLayer(X)
                    if not transfer or dropout:
                        X = dropoutLayer(X)

                    dnn = keras.layers.Dense(units=self.L,
                                             name = 'accDense2',
                                             kernel_initializer = kernel_initializer)

                    bnLayer = keras.layers.BatchNormalization(name = 'accBatch6')
                    activationLayer = keras.layers.ReLU()
                    dropoutLayer = keras.layers.Dropout(rate=0.25)

                    X = dnn(X)
                    X = bnLayer(X)
                    X = activationLayer(X)
                    if not transfer or dropout:
                        X = dropoutLayer(X)


                acc_encoding = X

            return keras.models.Model(inputs=inputAcc,
                                      outputs=acc_encoding,
                                      name='AccelerationEncoder')


        elif useFFT:
            if fusion == 'Depth':
                _, channels = inputAccShape
                bnLayer = keras.layers.BatchNormalization(name='accBatch1')
                X = bnLayer(X)

                paddingLayer = keras.layers.ZeroPadding1D(padding=1)
                conv1D = keras.layers.Conv1D(
                    filters=channels * 16,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    name='accConv1',
                    kernel_initializer=kernel_initializer
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
                    name='accConv2',
                    kernel_initializer=kernel_initializer
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
                    name='accConv3',
                    kernel_initializer=kernel_initializer
                )

                X = conv1D(X)
                X = activationLayer(X)
                X = poolingLayer(X)

                flattenLayer = keras.layers.Flatten()
                dropoutLayer = keras.layers.Dropout(rate=self.args.train_args['input_dropout'])

                X = flattenLayer(X)

                if not transfer or dropout:
                    X = dropoutLayer(X)

                dnn = keras.layers.Dense(units=dimension,
                                         name='accDense1',
                                         kernel_initializer = kernel_initializer)

                bnLayer = keras.layers.BatchNormalization(name='accBatch2')
                activationLayer = keras.layers.ReLU()
                dropoutLayer = keras.layers.Dropout(rate=0.25)

                X = dnn(X)
                X = bnLayer(X)
                X = activationLayer(X)
                if not transfer or dropout:
                    X = dropoutLayer(X)

                dnn = keras.layers.Dense(units=self.L,
                                         name='accDense2',
                                         kernel_initializer = kernel_initializer)

                bnLayer = keras.layers.BatchNormalization(name='accBatch3')
                activationLayer = keras.layers.ReLU()
                dropoutLayer = keras.layers.Dropout(rate=0.25)

                X = dnn(X)
                X = bnLayer(X)
                X = activationLayer(X)
                if not transfer or dropout:
                    X = dropoutLayer(X)

            acc_encoding = X

            return keras.models.Model(inputs=inputAcc,
                                      outputs=acc_encoding,
                                      name='AccelerationEncoder')

        else:
            if fusion == 'Depth':
                _, channels = inputAccShape
                bnLayer = keras.layers.BatchNormalization(name='accBatch1')
                X = bnLayer(X)


                paddingLayer = keras.layers.ZeroPadding1D(padding=1)
                conv1D = keras.layers.Conv1D(
                    filters=channels * 16,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    name='accConv1',
                    kernel_initializer=kernel_initializer
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
                    name='accConv2',
                    kernel_initializer=kernel_initializer
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
                    name='accConv3',
                    kernel_initializer=kernel_initializer
                )

                X = conv1D(X)
                X = activationLayer(X)
                X = poolingLayer(X)

                flattenLayer = keras.layers.Flatten()
                dropoutLayer = keras.layers.Dropout(rate=self.args.train_args['input_dropout'])

                X = flattenLayer(X)

                if not transfer or dropout:
                    X = dropoutLayer(X)

                dnn = keras.layers.Dense(units=256,
                                         name='accDense1',
                                         kernel_initializer = kernel_initializer)

                bnLayer = keras.layers.BatchNormalization(name='accBatch2')

                activationLayer = keras.layers.ReLU()
                dropoutLayer = keras.layers.Dropout(rate=0.25)

                X = dnn(X)
                X = bnLayer(X)
                X = activationLayer(X)
                if not transfer or dropout:
                    X = dropoutLayer(X)

                dnn = keras.layers.Dense(units=self.L,
                                         name='accDense2',
                                         kernel_initializer = kernel_initializer)

                bnLayer = keras.layers.BatchNormalization(name='accBatch3')

                activationLayer = keras.layers.ReLU()
                dropoutLayer = keras.layers.Dropout(rate=0.25)

                X = dnn(X)
                X = bnLayer(X)
                X = activationLayer(X)
                if not transfer or dropout:
                    X = dropoutLayer(X)

            acc_encoding = X

            return keras.models.Model(inputs = inputAcc,
                                      outputs = acc_encoding,
                                      name = 'AccelerationEncoder')

    def get_loc_encoder(self,
                        kernel_initializer = keras.initializers.he_uniform(),
                        transfer = False, dropout = False):

        inputLocSignalsShape = list(self.input_shapes[1])[1:]
        inputLocFeaturesShape = list(self.input_shapes[2])[1:]
        inputLocSignals = keras.Input( shape = (*inputLocSignalsShape, ) )
        inputLocFeatures = keras.Input(shape=(*inputLocFeaturesShape,))

        fusion = self.args.train_args['loc_fusion']


        if fusion == 'DNN':
            bnLayer = keras.layers.BatchNormalization()
            X = bnLayer(inputLoc)
            dropoutLayer = keras.layers.Dropout(rate=self.args.train_args['input_dropout'])
            X = dropoutLayer(X)


            denseLayer = keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer
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
                kernel_initializer=kernel_initializer
            )

            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.Activation('selu')
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)


            denseLayer = keras.layers.Dense(
                units=self.L,
                kernel_initializer=kernel_initializer
            )

            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.Activation('selu')

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)



            loc_encodings = X

            return keras.models.Model(inputs = inputLoc,
                                      outputs = loc_encodings,
                                      name = 'LocationEncoder')

        elif fusion in ['LSTM', 'BidirectionalLSTM', 'CNNLSTM', 'FCLSTM']:
            nullLoc = self.args.train_args['nullLoc']



            X = inputLocSignals


            if nullLoc == 'masking':
                masking_layer = keras.layers.Masking(mask_value = self.mask, name = 'maskLayer1')
                X = masking_layer(X)

            bnLayer = keras.layers.BatchNormalization(name = 'locBatch')
            X = bnLayer(X)

            if fusion == 'FCLSTM':
                dropoutLayer = keras.layers.Dropout(rate=0.25)
                X = dropoutLayer(X)

                TDDenseLayer = TimeDistributed(
                    keras.layers.Dense(
                        units=32,
                        kernel_initializer=kernel_initializer,
                        name='TDlocDense1'
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
                        kernel_initializer=kernel_initializer,
                        name='TDlocDense2'
                    )
                )
                TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch2')
                TDactivationLayer = MaskRelu()


                X = TDDenseLayer(X)
                X = TDbnLayer(X)
                X = TDactivationLayer(X)

                lstmLayer1 = tf.keras.layers.LSTM(units=64,
                                                  return_sequences=True,
                                                  name='locLSTM1')
                lstmLayer2 = tf.keras.layers.LSTM(units=64,
                                                  name='locLSTM2')

                X = lstmLayer1(X)
                X = lstmLayer2(X)

            else:
                if fusion == 'BidirectionalLSTM':
                    lstmLayer = Bidirectional(tf.keras.layers.LSTM(units = 128),
                                              name = 'locBiLSTM')

                elif fusion == 'CNNLSTM':
                    cnnLayer = keras.layers.Conv2D(
                        filters=1,
                        kernel_size=(1,inputLocSignalsShape[1]),
                        strides=1,
                        padding='valid',
                        name='locCNN',
                        kernel_initializer=kernel_initializer
                    )

                    X = cnnLayer(X)
                    X = tf.squeeze(X, axis=-1)

                    lstmLayer = tf.keras.layers.LSTM(units = 256,
                                                     name = 'locLSTM')

                else:


                    lstmLayer = tf.keras.layers.LSTM(units = 128,
                                                     name = 'locLSTM')

                X = lstmLayer(X)


            dropoutLayer = keras.layers.Dropout(rate=self.args.train_args['input_dropout'])

            if not transfer or dropout:
                X = dropoutLayer(X)

            X = tf.concat([X, inputLocFeatures], axis=1)


            denseLayer = keras.layers.Dense(
                units=128,
                kernel_initializer=kernel_initializer,
                name = 'locDense1'
            )

            bnLayer = keras.layers.BatchNormalization(name = 'locBatch1')
            activationLayer = MaskRelu()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)

            if not transfer or dropout:
                X = dropoutLayer(X)


            denseLayer = keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer,
                name = 'locDense2'
            )

            bnLayer = keras.layers.BatchNormalization(name = 'locBatch2')
            activationLayer = MaskRelu()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)

            if not transfer or dropout:
                X = dropoutLayer(X)


            denseLayer = keras.layers.Dense(
                units=self.L,
                kernel_initializer=kernel_initializer,
                name = 'locDense3'
            )

            bnLayer = keras.layers.BatchNormalization(name = 'locBatch3')
            activationLayer = MaskRelu()
            dropoutLayer = keras.layers.Dropout(rate=0.25)

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)

            if not transfer or dropout:
                X = dropoutLayer(X)

            mask_w = K.switch(
                tf.reduce_all(tf.equal(inputLocFeatures, self.mask), axis=1, keepdims=True),
                lambda: tf.zeros_like(X), lambda: tf.ones_like(X)
            )
            loc_encodings = tf.multiply(X, mask_w)


            return keras.models.Model(inputs = [inputLocSignals, inputLocFeatures],
                                      outputs = loc_encodings,
                                      name = 'LocationEncoder')

    def get_attention_layer(self,
                            kernel_initializer = keras.initializers.glorot_normal(),
                            kernel_regularizer = keras.regularizers.l2(0.01),
                            data = None):


        encodings = keras.Input(shape = (self.L))
        D_layer = keras.layers.Dense(units=self.D,
                                     activation='tanh',
                                     kernel_initializer = kernel_initializer,
                                     kernel_regularizer = kernel_regularizer)

        if self.gated:
            G_layer = keras.layers.Dense(units=self.D,
                                         activation='sigmoid',
                                         kernel_initializer = kernel_initializer,
                                         kernel_regularizer = kernel_regularizer)

        K_layer = keras.layers.Dense(units=self.K,
                                     kernel_initializer = kernel_initializer)

        attention_weights = D_layer(encodings)

        if self.gated:
            attention_weights = attention_weights * G_layer(encodings)

        attention_weights = K_layer(attention_weights)

        if not data:
            return keras.models.Model(inputs = encodings ,
                                  outputs = attention_weights,
                                  name = 'AttentionLayer')

        else:
            if data == 'acceleration':
                return keras.models.Model(inputs=encodings,
                                          outputs=attention_weights,
                                          name='AccelerationAttentionLayer')

            elif data == 'location':
                return keras.models.Model(inputs=encodings,
                                          outputs=attention_weights,
                                          name='LocationAttentionLayer')

    def get_classifier(self,
                       kernel_initializer = keras.initializers.glorot_normal()):

        if self.fusion == 'MIL':
            if not self.seperateMIL:
                pooling = keras.Input(shape=(self.L))

            else:
                pooling = keras.Input(shape=(2 * self.L))

        if self.fusion == 'concat':
            pooling = keras.Input(shape=(2 * self.L))


        X = pooling


        if self.args.train_args['classifier_layers']:

            denseLayer = keras.layers.Dense(
                units=self.L // 2,
                kernel_initializer=keras.initializers.he_uniform()
            )

            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.ReLU()

            X = denseLayer(X)
            X = bnLayer(X)
            X = activationLayer(X)


        if self.args.train_args['drop_run']:
            finalLayer = keras.layers.Dense(units=7,
                                            activation='softmax',
                                            kernel_initializer=kernel_initializer)


        else:
            finalLayer = keras.layers.Dense(units=8,
                                            activation='softmax',
                                            kernel_initializer=kernel_initializer)


        y_pred = finalLayer(X)


        return keras.models.Model(inputs = pooling,
                                  outputs = y_pred,
                                  name = 'Classifier')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_tracker
        ]

    def forward_pass(self, acc_bags, loc_signals_bags, loc_features_bags, return_weights = False, batchSize = None):

        accShape = self.input_shapes[0]
        accBagSize = list(accShape)[0]
        accSize = list(accShape)[1:]

        locSignalsShape = self.input_shapes[1]
        locSignalsBagSize = list(locSignalsShape)[0]
        locSignalsSize = list(locSignalsShape)[1:]

        locFeaturesShape = self.input_shapes[2]
        locFeaturesBagSize = list(locFeaturesShape)[0]
        locFeaturesSize = list(locFeaturesShape)[1:]


        if self.args.train_args['padding_method'] == 'variableLength':
            locSignalsSize[0] = -1
            locFeaturesSize[0] = -1


        concAccBags = tf.reshape(acc_bags, (batchSize*accBagSize, *accSize))
        concLocSignalsBags = tf.reshape(loc_signals_bags, (batchSize*locSignalsBagSize, *locSignalsSize))
        concLocFeaturesBags = tf.reshape(loc_features_bags, (batchSize * locFeaturesBagSize, *locFeaturesSize))

        accEncodings = self.acc_encoder(concAccBags)
        locEncodings = self.loc_encoder([concLocSignalsBags,concLocFeaturesBags])

        if self.fusion == 'MIL':
            if not self.seperateMIL:
                accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, self.L])
                locEncodings = tf.reshape(locEncodings, [batchSize, locSignalsBagSize, self.L])
                Encodings = tf.keras.layers.concatenate([accEncodings, locEncodings], axis=-2)
                Encodings = tf.reshape(Encodings, (batchSize*(accBagSize + locSignalsBagSize),self.L))

                attention_weights = self.attention_layer(Encodings)
                attention_weights = tf.reshape(attention_weights,[batchSize,accBagSize+locSignalsBagSize])
                softmax = keras.layers.Softmax()
                attention_weights = tf.expand_dims(softmax(attention_weights), -2)

                Encodings = tf.reshape(Encodings, [batchSize, accBagSize + locSignalsBagSize, self.L])

                if batchSize == 1:
                    pooling = tf.expand_dims(tf.squeeze(tf.matmul(attention_weights, Encodings)),axis= 0)

                else:
                    pooling = tf.squeeze(tf.matmul(attention_weights,Encodings))

            else:
                acc_attention_weights = self.acc_attention_layer(accEncodings)
                acc_attention_weights = tf.reshape(acc_attention_weights,
                                                   [batchSize,accBagSize])

                loc_attention_weights = self.loc_attention_layer(locEncodings)
                loc_attention_weights = tf.reshape(loc_attention_weights,
                                                   [batchSize,locSignalsBagSize])

                softmax = keras.layers.Softmax()

                acc_attention_weights = tf.expand_dims(softmax(acc_attention_weights), -2)
                loc_attention_weights = tf.expand_dims(softmax(loc_attention_weights), -2)

                accEncodings = tf.reshape(accEncodings, [batchSize,accBagSize,self.L])
                locEncodings = tf.reshape(locEncodings, [batchSize,locSignalsBagSize,self.L])

                if batchSize == 1:
                    accPooling = tf.expand_dims(tf.squeeze(tf.matmul(acc_attention_weights,accEncodings)),axis= 0)
                else:
                    accPooling = tf.squeeze(tf.matmul(acc_attention_weights,accEncodings))

                if batchSize == 1:
                    locPooling = tf.expand_dims(tf.squeeze(tf.matmul(loc_attention_weights,locEncodings)),axis= 0)
                else:
                    locPooling = tf.squeeze(tf.matmul(loc_attention_weights,locEncodings))


                pooling = tf.keras.layers.concatenate([accPooling,locPooling])

        elif self.fusion == 'concat':
            accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, self.L])
            locEncodings = tf.reshape(locEncodings, [batchSize, locSignalsBagSize, self.L])
            pooling = tf.squeeze(tf.keras.layers.concatenate([accEncodings, locEncodings], axis=-1))



        y_pred = self.classifier(pooling)

        if return_weights and self.fusion == 'MIL':
            if not self.seperateMIL:
                return attention_weights, y_pred

            else:
                return acc_attention_weights, loc_attention_weights, y_pred

        return y_pred

    def train_step(self, data):
        bags, y_true = data
        acc_bags, loc_signals_bags, loc_features_bags = bags

        with tf.GradientTape() as tape:

            y_pred = self.forward_pass(acc_bags,
                                       loc_signals_bags,
                                       loc_features_bags,
                                       batchSize=self.batchSize)

            loss = self.loss_estimator(y_true, y_pred)

        if not self.seperateMIL:
            learnable_weights = self.classifier.trainable_weights

            if self.fusion == 'MIL':
                learnable_weights = self.attention_layer.trainable_weights + learnable_weights

            if not self.transferLearningAcc:
                learnable_weights = self.acc_encoder.trainable_weights + learnable_weights

            if not self.transferLearningLoc:
                learnable_weights = self.loc_encoder.trainable_weights + learnable_weights



        grads = tape.gradient(
            loss,
            learnable_weights
        )

        self.optimizer.apply_gradients(
            zip(
                grads,
                learnable_weights
            )
        )

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y_true,y_pred)


        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        bags , y_true = data
        acc_bags , loc_signals_bags, loc_features_bags = bags

        y_pred = self.forward_pass(acc_bags,
                                   loc_signals_bags,
                                   loc_features_bags,
                                   batchSize=self.valBatchSize)

        loss = self.loss_estimator(y_true, y_pred)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y_true,y_pred)


        return {m.name: m.result() for m in self.metrics}

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False, return_weights = False):


        batchSize = inputs[0].shape[0]


        if return_weights:
            if not self.seperateMIL:
                attention_weights,pred = self.forward_pass(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    return_weights=True,
                    batchSize=batchSize
                )

                attention_weights = tf.squeeze(attention_weights)
                return attention_weights,pred

            else:
                acc_attention_weights, loc_attention_weights, pred = self.forward_pass(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    return_weights=True,
                    batchSize=batchSize
                )

                acc_attention_weights = tf.squeeze(acc_attention_weights)
                loc_attention_weights = tf.squeeze(loc_attention_weights)

                return acc_attention_weights, loc_attention_weights, pred

        else:
            pred = self.forward_pass(
                inputs[0],
                inputs[1],
                inputs[2],
                batchSize=batchSize
            )

            return pred

import tensorflow.keras.backend as K
def get_loss_function(Wp,Wn):

    def weighted_categorical_crossentropy(y_true,y_pred):


        first_term = Wp * y_true * tf.math.log(tf.clip_by_value(y_pred,1e-15,1.0))
        second_term = Wn * (1. - y_true) * tf.math.log(tf.clip_by_value(1.-y_pred,1e-15,1.))

        loss = -tf.reduce_sum(first_term + second_term, axis=-1)

        return loss

    return weighted_categorical_crossentropy


def get_focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def MIL_fit(SignalsDataset,
            evaluation = True,
            summary = False,
            verbose = 1):

    postprocessing = SignalsDataset.shl_args.train_args['postprocessing']
    postprocessingMethod = SignalsDataset.shl_args.train_args['postprocessing_method']
    decisionTree = SignalsDataset.shl_args.train_args['decision_tree']

    if decisionTree:

        trainX, trainY, valX, valY, testX, testY = SignalsDataset(randomTree=True,DHMM=False)


        depths = []
        for i in range(3, 20):
            clf = tree.DecisionTreeClassifier(max_depth=i)
            clf = clf.fit(trainX, trainY)
            depths.append((i, clf.score(valX, valY)))
            # print(clf.score(testX, testY))

        # print(depths)
        max_score = 0
        for depth in depths:
            if depth[1]>max_score:
                best_depth = depth[0]
                max_score = depth[1]


        # train_X = pd.concat([trainX,valX])
        # train_Y = pd.concat([trainY,valY])

        clf = tree.DecisionTreeClassifier(max_depth=best_depth)
        clf.fit(trainX,trainY)
        print(clf.score(testX,testY))


        trainX, trainY, testX, testY, trans_mx = SignalsDataset(randomTree=True, DHMM=True)

        valY_ = clf.predict(valX)

        conf_mx = sklearn.metrics.confusion_matrix(valY_,valY)

        discrete_model = hmm.MultinomialHMM(n_components=5,
                           algorithm='viterbi',  # decoder algorithm.
                           random_state=93,
                           n_iter=10
                           )

        print(trans_mx)
        print(conf_mx)

        discrete_model.startprob_ = [1./5. for _ in range(5)]
        discrete_model.transmat_ = trans_mx
        discrete_model.emissionprob_ = conf_mx



        X = []
        Y = []
        lengths = []
        for seq,seq_y in zip(testX,testY):
            # print(seq_y)
            # print(seq)
            # print(clf.predict_proba(seq))
            X.extend(clf.predict_proba(seq))
            lengths.append(len(seq))
            Y.extend(seq_y['label'].to_list())


        Y_ = discrete_model.predict(X,lengths = lengths)
        score = sklearn.metrics.accuracy_score(Y,Y_)

        print(score)






    elif postprocessing:

        if postprocessingMethod == 'LSTM':
            postprocess_Model = keras.Sequential()

            for postprocess in [True,False]:

                rounds = [True,False] if postprocess else [True]
                for round in rounds:
                    L = 256
                    D = 128


                    if SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'train':

                        saved_model_loc = locTransferLearning.fit(
                            evaluation=evaluation,
                            summary=summary,
                            verbose=verbose,
                            L=L
                        )

                    elif SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'load':

                        save_dir = os.path.join('training', 'saved_models')
                        if not os.path.isdir(save_dir):
                            return

                        model_type = 'location_classifier'
                        model_name = 'shl_%s_model.h5' % model_type
                        filepath = os.path.join(save_dir, model_name)

                        saved_model_loc = filepath

                    else:

                        saved_model_loc = None


                    if SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'train':

                        saved_model_acc = transferLearning.fit(
                            evaluation = evaluation,
                            L = L,
                            summary = summary,
                            verbose = verbose,
                            use_simCLR = SignalsDataset.shl_args.train_args['simCLR'],
                            postprocess = postprocess,
                            round = round
                        )

                    elif SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'load':
                        save_dir = os.path.join('training', 'saved_models')
                        if not os.path.isdir(save_dir):
                            return

                        model_type = 'acceleration_classifier'
                        model_name = 'shl_%s_model.h5' % model_type
                        filepath = os.path.join(save_dir, model_name)
                        saved_model_acc = filepath

                    else:

                        saved_model_acc = None

                    fusion = SignalsDataset.shl_args.train_args['fusion']
                    train , val , test = SignalsDataset(postprocess = postprocess, round = round)

                    user = SignalsDataset.shl_args.train_args['test_user']

                    logdir = os.path.join('logs_user' + str(user),'MIL_tensorboard')

                    try:
                        shutil.rmtree(logdir)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

                    val_steps = SignalsDataset.valSize // SignalsDataset.valBatchSize

                    if SignalsDataset.padding_method == 'variableLength':
                        train_steps = SignalsDataset.batches

                    else:
                        train_steps = SignalsDataset.trainSize // SignalsDataset.trainBatchSize

                    test_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize



                    if SignalsDataset.shl_args.train_args['loss_function'] == 'weighted':

                        N = SignalsDataset.trainSize

                        lb_count = np.zeros(SignalsDataset.n_labels)

                        for index in SignalsDataset.train_indices:
                            lb_count = lb_count + SignalsDataset.lbsTfrm(SignalsDataset.labels[index, 0])

                        Wp = tf.convert_to_tensor(N / (2. * lb_count), tf.float32)
                        Wn = tf.convert_to_tensor(N / (2. * ( N - lb_count )), tf.float32)

                        loss_function = get_loss_function(Wp, Wn)

                    elif SignalsDataset.shl_args.train_args['loss_function'] == 'focal':
                        loss_function = get_focal_loss()

                    else:
                        loss_function = keras.losses.CategoricalCrossentropy()



                    Model = newMILattention(input_shapes = SignalsDataset.inputShape,
                                            args = SignalsDataset.shl_args,
                                            loss_function = loss_function,
                                            L = L, D = D,
                                            acc_weights = saved_model_acc,
                                            loc_weights = saved_model_loc,
                                            fusion = fusion)


                    Model.compile(
                        optimizer = keras.optimizers.Adam(learning_rate=SignalsDataset.shl_args.train_args['learning_rate'])
                    )

                    val_metrics = valMetrics(val,
                                             SignalsDataset.valBatchSize,
                                             val_steps)

                    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
                    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
                    w_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_val')
                    w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')

                    if fusion == 'MIL':
                        val_cm = confusion_metric(val,
                                                  SignalsDataset.valBatchSize,
                                                  SignalsDataset.shl_args.train_args['accBagSize'],
                                                  SignalsDataset.shl_args.train_args['locBagSize'],
                                                  val_steps,
                                                  file_writer_val,
                                                  w_file_writer_val,
                                                  drop_run = SignalsDataset.shl_args.train_args['drop_run'])

                    save_dir = os.path.join('training','saved_models')
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)


                    model_type = 'MILattention'
                    model_name = 'shl_%s_model.h5' %model_type
                    filepath = os.path.join(save_dir, model_name)

                    save_model = keras.callbacks.ModelCheckpoint(
                                filepath = filepath,
                                monitor = 'val_Loss',
                                verbose = 1,
                                save_best_only = True,
                                mode = 'min',
                                save_weights_only=True
                        )

                    early_stopping = keras.callbacks.EarlyStopping(
                        monitor = 'val_Loss',
                        min_delta = 0,
                        patience = 30,
                        mode = 'min',
                        verbose = 1
                    )

                    reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
                        monitor = 'val_Loss',
                        factor = 0.4,
                        patience = 10,
                        verbose = 1,
                        mode = 'min'
                    )

                    if summary:
                        print(Model.summary())

                    if fusion == 'MIL':
                        callbacks = [tensorboard_callback,
                                     val_metrics,val_cm,
                                     save_model,
                                     early_stopping,
                                     reduce_lr_plateau]

                    elif fusion == 'concat':
                        callbacks = [tensorboard_callback,
                                     val_metrics,
                                     save_model,
                                     early_stopping,
                                     reduce_lr_plateau]

                    Model.fit(
                        train,
                        epochs=SignalsDataset.shl_args.train_args['epochs'],
                        steps_per_epoch = train_steps,
                        validation_data = val,
                        validation_steps = val_steps,
                        callbacks = callbacks,
                        use_multiprocessing=True,
                        verbose=verbose
                    )

                    if SignalsDataset.padding_method == 'variableLength':
                        Model.built = True



                    Model.load_weights(filepath)
                    Model.acc_encoder.trainable = True
                    Model.loc_encoder.trainable = True
                    Model.save_weights(filepath)

                    if evaluation:

                        test_metrics = testMetrics(test,SignalsDataset.testBatchSize,test_steps)

                        if fusion == 'MIL':
                            test_cm = testConfusionMetric(test,
                                                          SignalsDataset.testBatchSize,
                                                          SignalsDataset.shl_args.train_args['accBagSize'],
                                                          SignalsDataset.shl_args.train_args['locBagSize'],
                                                          test_steps,
                                                          file_writer_test,
                                                          w_file_writer_test,
                                                          drop_run = SignalsDataset.shl_args.train_args['drop_run'])

                        if fusion == 'MIL':
                            callbacks = [test_metrics, test_cm]

                        elif fusion == 'concat':
                            callbacks = [test_metrics]
                        Model.evaluate(test,steps=test_steps,callbacks=callbacks)

                    if postprocess:

                        train_x, train_y, val_x, val_y, test_x, test_y = SignalsDataset.get_seq_lbs(Model)


                        train_steps = len(train_x)
                        val_steps = len(val_x)
                        test_steps = len(test_x)

                        train = SignalsDataset.to_seq_generator(train_x, train_y)
                        val = SignalsDataset.to_seq_generator(val_x, val_y)
                        test = SignalsDataset.to_seq_generator(test_x, test_y)

                        train = SignalsDataset.seqs_batch_and_prefetch(train)
                        val = SignalsDataset.seqs_batch_and_prefetch(val)
                        test = SignalsDataset.seqs_batch_and_prefetch(test)

                        postprocess_Model = postprocessModel(
                            input_shapes=[[None, 8], [None, 8]]
                        )

                        postprocess_Model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=0.001)
                        )

                        # save_model = keras.callbacks.ModelCheckpoint(
                        #     filepath=filepath,
                        #     monitor='val_loss',
                        #     verbose=1,
                        #     save_best_only=True,
                        #     mode='min',
                        #     save_weights_only=True
                        # )

                        early_stopping = keras.callbacks.EarlyStopping(
                            monitor='val_Loss',
                            min_delta=0,
                            patience=30,
                            mode='min',
                            verbose=1
                        )

                        reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
                            monitor='val_Loss',
                            factor=0.4,
                            patience=10,
                            verbose=1,
                            mode='min'
                        )

                        print(postprocess_Model.summary())

                        postprocess_Model.fit(
                            train,
                            validation_data=val,
                            epochs=80,
                            steps_per_epoch=train_steps,
                            validation_steps=val_steps,
                            verbose=True,
                            use_multiprocessing=True,
                            callbacks=[
                                reduce_lr_plateau,
                                early_stopping
                            ]
                        )

                        postprocess_Model.evaluate(
                            test,
                            steps=test_steps,
                            batch_size=1,
                            verbose=True,
                            use_multiprocessing=True
                        )




                    finetuning = SignalsDataset.shl_args.train_args['finetuning']

                    if finetuning:

                        fn_lr = SignalsDataset.shl_args.train_args['finetuning_learning_rate']
                        fn_epochs = SignalsDataset.shl_args.train_args['finetuning_epochs']

                        logdir = os.path.join('logs_user' + str(user), 'finetuning_MIL_tensorboard')

                        try:
                            shutil.rmtree(logdir)
                        except OSError as e:
                            print("Error: %s - %s." % (e.filename, e.strerror))

                        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

                        val_metrics = valMetrics(val,
                                                 SignalsDataset.valBatchSize,
                                                 val_steps)

                        file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
                        file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')

                        val_cm = confusion_metric(val,
                                                  SignalsDataset.valBatchSize,
                                                  val_steps,
                                                  file_writer_val,
                                                  w_file_writer_val,
                                                  drop_run=SignalsDataset.shl_args.train_args['drop_run'])

                        Model.acc_encoder.trainable = True
                        Model.loc_encoder.trainable = True

                        Model.compile(
                            optimizer = keras.optimizers.Adam(learning_rate = fn_lr)
                        )

                        Model.fit(
                            train,
                            epochs = fn_epochs,
                            steps_per_epoch = train_steps,
                            validation_data = val,
                            validation_steps = val_steps,
                            callbacks = [tensorboard_callback,
                                         val_metrics,val_cm,
                                         save_model],
                            use_multiprocessing=True,
                            verbose=verbose
                        )

                        Model.load_weights(filepath)

                        if evaluation:

                            test_cm = testConfusionMetric(test,
                                                          SignalsDataset.testBatchSize,
                                                          test_steps,
                                                          file_writer_test,
                                                          w_file_writer_test,
                                                          drop_run=SignalsDataset.shl_args.train_args['drop_run'])

                            Model.evaluate(test, steps=test_steps, callbacks=[test_cm])

            return Model, postprocess_Model

        elif postprocessingMethod == 'Polynomial':
            trans_mx = pd.DataFrame(
                        np.zeros(shape=(SignalsDataset.n_labels,SignalsDataset.n_labels))
                    )
            conf_mx = pd.DataFrame(
                        np.zeros(shape=(SignalsDataset.n_labels,SignalsDataset.n_labels))
                    )
            for user_seperated in [True,False]:
                rounds = [True,False] if user_seperated else True
                for round in rounds:
                    L = 256
                    D = 128

                    if SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'train':

                        saved_model_loc = locTransferLearning.fit(
                            evaluation=evaluation,
                            summary=summary,
                            verbose=verbose,
                            L=L
                        )

                    elif SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'load':

                        save_dir = os.path.join('training', 'saved_models')
                        if not os.path.isdir(save_dir):
                            return

                        model_type = 'location_classifier'
                        model_name = 'shl_%s_model.h5' % model_type
                        filepath = os.path.join(save_dir, model_name)

                        saved_model_loc = filepath

                    else:

                        saved_model_loc = None

                    if SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'train':

                        saved_model_acc = transferLearning.fit(
                            evaluation=evaluation,
                            L=L,
                            summary=summary,
                            verbose=verbose,
                            use_simCLR=SignalsDataset.shl_args.train_args['simCLR'],
                            user_seperated=user_seperated,
                            round = round
                        )

                    elif SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'load':
                        save_dir = os.path.join('training', 'saved_models')
                        if not os.path.isdir(save_dir):
                            return

                        model_type = 'acceleration_classifier'
                        model_name = 'shl_%s_model.h5' % model_type
                        filepath = os.path.join(save_dir, model_name)
                        saved_model_acc = filepath

                    else:

                        saved_model_acc = None

                    fusion = SignalsDataset.shl_args.train_args['fusion']
                    train, val, test = SignalsDataset(user_seperated = user_seperated, round = round)

                    user = SignalsDataset.shl_args.train_args['test_user']

                    logdir = os.path.join('logs_user' + str(user), 'MIL_tensorboard')

                    try:
                        shutil.rmtree(logdir)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

                    val_steps = SignalsDataset.valSize // SignalsDataset.valBatchSize

                    if SignalsDataset.padding_method == 'variableLength':
                        train_steps = SignalsDataset.batches

                    else:
                        train_steps = SignalsDataset.trainSize // SignalsDataset.trainBatchSize

                    test_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize

                    if SignalsDataset.shl_args.train_args['loss_function'] == 'weighted':

                        N = SignalsDataset.trainSize

                        lb_count = np.zeros(SignalsDataset.n_labels)

                        for index in SignalsDataset.train_indices:
                            lb_count = lb_count + SignalsDataset.lbsTfrm(SignalsDataset.labels[index, 0])

                        Wp = tf.convert_to_tensor(N / (2. * lb_count), tf.float32)
                        Wn = tf.convert_to_tensor(N / (2. * (N - lb_count)), tf.float32)

                        loss_function = get_loss_function(Wp, Wn)

                    elif SignalsDataset.shl_args.train_args['loss_function'] == 'focal':
                        loss_function = get_focal_loss()

                    else:
                        loss_function = keras.losses.CategoricalCrossentropy()

                    Model = newMILattention(input_shapes=SignalsDataset.inputShape,
                                            args=SignalsDataset.shl_args,
                                            loss_function=loss_function,
                                            L=L, D=D,
                                            acc_weights=saved_model_acc,
                                            loc_weights=saved_model_loc,
                                            fusion=fusion)

                    Model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=SignalsDataset.shl_args.train_args['learning_rate'])
                    )

                    val_metrics = valMetrics(val,
                                             SignalsDataset.valBatchSize,
                                             val_steps)

                    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
                    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
                    w_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_val')
                    w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')

                    if fusion == 'MIL':
                        val_cm = confusion_metric(val,
                                                  SignalsDataset.valBatchSize,
                                                  SignalsDataset.shl_args.train_args['accBagSize'],
                                                  SignalsDataset.shl_args.train_args['locBagSize'],
                                                  val_steps,
                                                  file_writer_val,
                                                  w_file_writer_val,
                                                  drop_run=SignalsDataset.shl_args.train_args['drop_run'])

                    save_dir = os.path.join('training', 'saved_models')
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    model_type = 'MILattention'
                    model_name = 'shl_%s_model.h5' % model_type
                    filepath = os.path.join(save_dir, model_name)

                    save_model = keras.callbacks.ModelCheckpoint(
                        filepath=filepath,
                        monitor='val_Loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True
                    )

                    early_stopping = keras.callbacks.EarlyStopping(
                        monitor='val_Loss',
                        min_delta=0,
                        patience=30,
                        mode='min',
                        verbose=1
                    )

                    reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
                        monitor='val_Loss',
                        factor=0.4,
                        patience=10,
                        verbose=1,
                        mode='min'
                    )

                    if summary:
                        print(Model.summary())

                    if fusion == 'MIL':
                        callbacks = [tensorboard_callback,
                                     val_metrics, val_cm,
                                     save_model,
                                     early_stopping,
                                     reduce_lr_plateau]

                    elif fusion == 'concat':
                        callbacks = [tensorboard_callback,
                                     val_metrics,
                                     save_model,
                                     early_stopping,
                                     reduce_lr_plateau]

                    Model.fit(
                        train,
                        epochs=SignalsDataset.shl_args.train_args['epochs'],
                        steps_per_epoch=train_steps,
                        validation_data=val,
                        validation_steps=val_steps,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        verbose=verbose
                    )

                    if SignalsDataset.padding_method == 'variableLength':
                        Model.built = True

                    Model.load_weights(filepath)
                    Model.acc_encoder.trainable = True
                    Model.loc_encoder.trainable = True
                    Model.save_weights(filepath)

                    if evaluation:

                        test_metrics = testMetrics(test, SignalsDataset.testBatchSize, test_steps)

                        if fusion == 'MIL':
                            test_cm = testConfusionMetric(test,
                                                          SignalsDataset.testBatchSize,
                                                          SignalsDataset.shl_args.train_args['accBagSize'],
                                                          SignalsDataset.shl_args.train_args['locBagSize'],
                                                          test_steps,
                                                          file_writer_test,
                                                          w_file_writer_test,
                                                          drop_run=SignalsDataset.shl_args.train_args['drop_run'])

                        if fusion == 'MIL':
                            callbacks = [test_metrics, test_cm]

                        elif fusion == 'concat':
                            callbacks = [test_metrics]
                        Model.evaluate(test, steps=test_steps, callbacks=callbacks)

                    if user_seperated:

                        trans_mx_,conf_mx_ = SignalsDataset.postprocess(Model=Model, postprocessing='Polynomial', fit=user_seperated)
                        trans_mx.add(trans_mx_)
                        conf_mx.add(conf_mx_)

                    else:

                        trans_mx["sum"] = trans_mx.sum(axis=1)
                        trans_mx = trans_mx.div(trans_mx["sum"], axis=0)
                        trans_mx = trans_mx.drop(columns=['sum'])
                        trans_mx = trans_mx.values.tolist()

                        conf_mx["sum"] = conf_mx.sum(axis=1)
                        conf_mx = conf_mx.div(conf_mx["sum"], axis=0)
                        conf_mx = conf_mx.drop(columns=['sum'])
                        conf_mx = conf_mx.values.tolist()

                        startprob = [1. / SignalsDataset.n_labels for _ in range(SignalsDataset.n_labels)]

                        x,y,lengths = SignalsDataset.postprocess(Model=Model, postprocessing=postprocessModel, fit=user_seperated)

                        discrete_model = hmm.MultinomialHMM(n_components=SignalsDataset.n_labels,
                                                            algorithm='viterbi',  # decoder algorithm.
                                                            random_state=93,
                                                            n_iter=10
                                                            )

                        print(trans_mx)
                        print(conf_mx)
                        print(startprob)

                        discrete_model.startprob_ = startprob
                        discrete_model.transmat_ = trans_mx
                        discrete_model.emissionprob_ = conf_mx

                        y_ = discrete_model.predict(x,lengths)
                        score = sklearn.metrics.accuracy_score(y, y_)

                        print(score)

            return Model

