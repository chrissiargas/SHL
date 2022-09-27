import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from newSignalDataset import SignalsDataset
from newMILattention import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os

from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense



class postprocessModel(keras.Model):
    def __init__(self,
                 input_shapes,
                 loss_function = tf.keras.losses.CategoricalCrossentropy(),
                 L = 8, D = 8,
                 gated = True, both = True):
        super(postprocessModel, self).__init__()
        self.both = both
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
        dropoutLayer = Dropout(dropout_)
        X = dropoutLayer(X)

        rec_layer = LSTM(self.L)
        X = rec_layer(X)
        output_X = dropoutLayer(X)

        return keras.models.Model(
            inputs = input_X,
            outputs = output_X,
            name = 'ForwardEncoder'
        )

    def get_back_encoder(self, dropout = True):
        dropout_ = 0.25 if dropout else 0.

        input_shape = list(self.input_shapes[0])
        input_X = keras.Input(shape=(*input_shape,))

        X = input_X
        dropoutLayer = Dropout(dropout_)
        X = dropoutLayer(X)

        rec_layer = LSTM(self.L)
        X = rec_layer(X)
        output_X = dropoutLayer(X)

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

        if self.both:
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

        else:
            pooling = forEncodings



        y_pred = self.classifier(pooling)


        return y_pred

    def train_step(self, data):
        x, y = data
        forward_x, backward_x = x

        with tf.GradientTape() as tape:
            y_ = self.forward_pass(forward_x, backward_x)

            loss = self.loss_estimator(y, y_)

        learnable_weights = self.classifier.trainable_weights + \
                            self.forward_encoder.trainable_weights

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



SD = SignalsDataset()



fusion = SD.shl_args.train_args['fusion']
train, val, test = SD(postprocess=True, round=True)



if SD.shl_args.train_args['loss_function'] == 'weighted':

    N = SD.trainSize

    lb_count = np.zeros(SD.n_labels)

    for index in SD.train_indices:
        lb_count = lb_count + SD.lbsTfrm(SD.labels[index, 0])

    Wp = tf.convert_to_tensor(N / (2. * lb_count), tf.float32)
    Wn = tf.convert_to_tensor(N / (2. * (N - lb_count)), tf.float32)

    loss_function = get_loss_function(Wp, Wn)

else:
    loss_function = keras.losses.CategoricalCrossentropy()




if SD.shl_args.train_args['transfer_learning_loc'] == 'load':

    save_dir = os.path.join('training', 'saved_models')


    model_type = 'location_classifier'
    model_name = 'shl_%s_model.h5' % model_type
    filepath = os.path.join(save_dir, model_name)

    saved_model_loc = filepath

else:

    saved_model_loc = None



if SD.shl_args.train_args['transfer_learning_acc'] == 'load':
    save_dir = os.path.join('training', 'saved_models')


    model_type = 'acceleration_classifier'
    model_name = 'shl_%s_model.h5' % model_type
    filepath = os.path.join(save_dir, model_name)
    saved_model_acc = filepath

else:

    saved_model_acc = None



L = 256
D = 128

Model = newMILattention(input_shapes=SD.inputShape,
                        args=SD.shl_args,
                        loss_function=loss_function,
                        L=L, D=D, fusion = fusion,
                        acc_weights=saved_model_acc,
                        loc_weights=saved_model_loc
                        )

Model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=SD.shl_args.train_args['learning_rate'])
)

save_dir = os.path.join('training', 'saved_models')
model_type = 'MILattention'
model_name = 'shl_%s_model.h5' % model_type
filepath = os.path.join(save_dir, model_name)



Model.built = True
Model.acc_encoder.trainable = True
Model.loc_encoder.trainable = True
Model.load_weights(filepath)

# test_steps = SD.testSize // SD.testBatchSize
# Model.evaluate(test,steps=test_steps)

train_x, train_y, val_x, val_y, test_x, test_y = SD.get_seq_lbs(Model)

# print(train_seqs)
# print(val_seqs)
# print(test_seqs)

train_steps = len(train_x)
val_steps = len(val_x)
test_steps = len(test_x)

train = SD.to_seq_generator(train_x, train_y)
val = SD.to_seq_generator(val_x, val_y)
test = SD.to_seq_generator(test_x, test_y)

train = SD.seqs_batch_and_prefetch(train)
val = SD.seqs_batch_and_prefetch(val)
test = SD.seqs_batch_and_prefetch(test)

for seq in test.take(2):
    x,y = seq
    x1,x2 = x
    print(x1)
    print(x2)
    print(y)
    print()

postprocess_Model = postprocessModel(
    input_shapes = [[None,8],[None,8]]
)

postprocess_Model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
)


# save_model = keras.callbacks.ModelCheckpoint(
#     filepath=filepath,
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True,
#     mode='min',
#     save_weights_only=True
# )

print(postprocess_Model.summary())

postprocess_Model.fit(
    train,
    validation_data=val,
    epochs=160,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=True,
    use_multiprocessing=True
)

postprocess_Model.evaluate(
    test,
    steps=test_steps,
    batch_size=1,
    verbose=True,
    use_multiprocessing=True
)

#
#
# for batch in test.take(1):
#     test_data_signals = batch[0][1]
#     test_data_features = batch[0][2]
#     inputLocFeatures_model = tf.keras.models.Model(inputs=Model.loc_encoder.input, outputs=Model.loc_encoder.layers[5].output)
#     intermediate_model = tf.keras.models.Model(inputs=Model.loc_encoder.input, outputs=Model.loc_encoder.layers[18].output)
#     maskw_model = tf.keras.models.Model(inputs=Model.loc_encoder.input, outputs=Model.loc_encoder.layers[19].output)
#     full_model = tf.keras.models.Model(inputs=Model.loc_encoder.input, outputs=Model.loc_encoder.output)
#
#     print({i: v for i, v in enumerate(full_model.layers)})
#     print()
#     print({i: v for i, v in enumerate(intermediate_model.layers)})
#     print()
#     locSignalsShape = test_data_signals.shape
#     locFeaturesShape = test_data_features.shape
#     test_data_signals = tf.reshape(test_data_signals, (-1, *locSignalsShape[2:]))
#     test_data_features = tf.reshape(test_data_features, (-1, *locFeaturesShape[2:]))
#
#     output = intermediate_model.call([test_data_signals, test_data_features])
#     f_output = full_model.call([test_data_signals, test_data_features])
#     inputLocFeatures = inputLocFeatures_model.call([test_data_signals, test_data_features])
#     mask_w1 = maskw_model.call([test_data_signals, test_data_features])
#     mask_w2 = tf.reduce_all(tf.equal(inputLocFeatures, Model.mask), axis=1, keepdims=True)
#     mask_w = K.switch(
#                 mask_w2,
#                 lambda: tf.zeros_like(output), lambda: tf.ones_like(output)
#             )
#     testing = tf.multiply(output, mask_w)
#
#     for sample_dt, sample_dt2, sample_op, sample_op2, f_op, m1, m2 in zip(test_data_signals, test_data_features, output, testing, f_output, mask_w1, mask_w2):
#         print(sample_dt)
#         print()
#         print(sample_dt2)
#         print()
#         print(m1)
#         print(f_op)
#         print()
#         print(m2)
#         print(sample_op2)
#         print()

# test_steps = (SD.testSize // SD.testBatchSize)
#
# Model.evaluate(test, steps=test_steps)
#
# steps = SD.testSize // SD.testBatchSize
# batchSize = SD.testBatchSize
# total = batchSize * steps
# step = 0
# test_predict = np.zeros((total))
# test_true = np.zeros((total))
# weights = np.zeros((total, 2))
# nulls = np.zeros((total))
#
#
# for batch in test.take(steps):
#     test_data = batch[0]
#     test_target = batch[1]
#     w, pred = Model.call(test_data, return_weights=True)
#     test_predict[step * batchSize: (step + 1) * batchSize] = \
#         np.argmax(np.asarray(pred), axis=1)
#     test_true[step * batchSize: (step + 1) * batchSize] = np.argmax(test_target, axis=1)
#
#     nulls[step * batchSize: (step + 1) * batchSize] = np.array(
#         [np.sum(np.count_nonzero(loc == -10000000, axis=2) != 0) for loc in test_data[1]]
#     )
#
#     weights[step * batchSize: (step + 1) * batchSize] = np.array([np.sum(w[:,:SD.shl_args.train_args['accBagSize']]),
#                                                                   np.sum(w[:,SD.shl_args.train_args['accBagSize']:])])
#
#     step += 1
#
#
# wm_pred = np.concatenate([test_predict[:, np.newaxis], nulls[:,np.newaxis], weights], axis=1)
# wm_pred_df = pd.DataFrame(
#     wm_pred,
#     columns=['class', 'nulls', 'accWeight', 'locWeight']
# )
# wm_pred_class = wm_pred_df[['class', 'accWeight', 'locWeight']].groupby(['class'], as_index=False).mean()
#
#
#
#
# wm_true = np.concatenate([test_true[:, np.newaxis], nulls[:,np.newaxis], weights], axis=1)
# wm_true_df = pd.DataFrame(
#     wm_true,
#     columns=['class', 'nulls', 'accWeight', 'locWeight']
# )
# wm_true_class = wm_true_df[['class', 'accWeight', 'locWeight']].groupby(['class'], as_index=False).mean()
#
#
# wm_pred_nulls = wm_pred_df.groupby(['class','nulls'], as_index=False).mean()
# wm_pred_nulls = pd.pivot_table(wm_pred_nulls, values="accWeight",index=["class"], columns=["nulls"], fill_value=0)
# wm_pred_null_count = wm_pred_df.groupby(['class','nulls']).size().to_frame(name = 'size').reset_index()
# wm_pred_null_count = pd.pivot_table(wm_pred_null_count, values="size",index=["class"], columns=["nulls"], fill_value=0)
# wm_pred_null_count = wm_pred_null_count.astype('float') / wm_pred_null_count.sum(axis=1)[:,np.newaxis]
#
# y_axis_labels = [
#                 'Still',
#                 'Walking',
#                 'Run',
#                 'Bike',
#                 'Car',
#                 'Bus',
#                 'Train',
#                 'Subway'
#             ]
#
# fig, (ax,ax2) = plt.subplots(ncols=2)
# fig.subplots_adjust(wspace=0.01)
# sns.heatmap(wm_pred_nulls, ax=ax, cbar = False, annot = True)
# fig.colorbar(ax.collections[0], ax=ax,location="left", use_gridspec=False, pad=0.2)
# sns.heatmap(wm_pred_null_count, ax=ax2, cbar = False, annot = True)
# fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.2)
# fig.suptitle('Weight Matrix')
# plt.ylabel('')
# plt.xlabel('nulls')
# ax2.yaxis.tick_right()
# ax.set_yticklabels(labels=y_axis_labels,
#                    rotation = 45)
#
# ax2.tick_params(rotation=0)
# plt.show()
