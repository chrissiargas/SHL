import random

import numpy as np

from configParser import Parser
from extractData import extractData
from preprocessData import preprocessData
import tensorflow as tf
from transformers import *
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding



class SignalsDataset:

    def __init__(self,
                 regenerate = False,
                 deleteFolders = False,
                 verbose = False
                 ):



        parser = Parser()
        self.shl_args = parser.get_args()

        self.verbose = verbose

        if not regenerate:
            exData = extractData(self.shl_args)

            if not exData.found:

                ppData = preprocessData(args=self.shl_args,
                                             verbose=verbose)

                _,_,_ = ppData()
                del ppData

                self.acceleration, \
                self.labels, \
                self.location = exData(delete_dst=deleteFolders,
                                             delete_tmp=deleteFolders,
                                             delete_final=deleteFolders)




            else:
                self.acceleration, \
                self.labels, \
                self.location = exData(delete_dst=deleteFolders,
                                             delete_tmp=deleteFolders,
                                             delete_final=deleteFolders)

            del exData


        else:
            ppData = preprocessData(args=self.shl_args,
                                         verbose=verbose,
                                         delete_dst=True,
                                         delete_tmp=True,
                                         delete_final=True,
                                         delete_filter=True)

            ppData()

            del ppData

            exData = extractData(self.shl_args)

            self.acceleration, \
            self.labels, \
            self.location = exData(delete_dst=deleteFolders,
                                         delete_tmp=deleteFolders,
                                         delete_final=deleteFolders)

            del exData

        self.bags = self.labels.shape[0]

    def to_bags(self):

        pos_names = ['Torso',
                      'Hips',
                      'Bag',
                      'Hand']

        n_labels = self.bags

        bag_map = {
            'acc_bags' : [[] for _ in range(n_labels)],
            'lbs_bags' : [i for i in range(n_labels)],
            'loc_bags' : {}
        }



        samples = self.acceleration.shape[0]


        acc_bag_size = self.shl_args.train_args['accBagSize']
        loc_bag_size = self.shl_args.train_args['locBagSize']


        contain_label = self.shl_args.train_args['containLabel']


        for i,label in enumerate(self.labels):
            sample_index = label[1]
            label_index = label[2]

            bag_user = label[-3]
            bag_day = label[-2]
            bag_time = label[-1]

            bag_map['acc_bags'][i].append(sample_index)

            step = 1
            while step <= (acc_bag_size-1) // 2:
                for moved_index in [sample_index + step , sample_index - step]:
                    if 0 <= moved_index < samples:
                        pair_acc = self.acceleration[moved_index][label_index]


                        if contain_label:
                            start_time = self.acceleration[moved_index][0][-1]
                            end_time = self.acceleration[moved_index][-1][-1]



                        if pair_acc[-3] == bag_user and pair_acc[-2] == bag_day:
                            if contain_label:
                                if start_time <= bag_time <= end_time:
                                    bag_map['acc_bags'][i].append(moved_index)

                            else:
                                bag_map['acc_bags'][i].append(moved_index)

                step += 1


        for pos_index,pos_name in enumerate(pos_names):
            offset = 0
            last = False
            bag_map['loc_bags'][pos_name] = [[] for _ in range(n_labels)]
            start = 0
            tmp_n = 0

            for i,label in enumerate(self.labels):

                if last and offset == tmp_n:
                    break

                bag_user = label[-3]
                bag_day = label[-2]
                bag_time = label[-1]

                if i==0 or bag_user != self.labels[i-1][-3] or \
                    bag_day != self.labels[i-1][-2]:

                    if bag_user == 3 and bag_day == 3:
                        last = True

                    start += tmp_n
                    tmp_location = self.select_location(bag_user,
                                         bag_day,
                                         pos_index,
                                         start)

                    offset = 0
                    tmp_n = tmp_location.shape[0]


                begin = offset
                while offset < tmp_n:
                    if bag_time < tmp_location[offset,0,-1]:
                        offset = begin
                        break

                    elif tmp_location[offset,0,-1] <= bag_time <= tmp_location[offset,-1,-1]:
                        bag_map['loc_bags'][pos_name][i].append(offset + start)
                        offset += 1

                    elif offset == tmp_n-1 or \
                    tmp_location[offset,-1,-1] < bag_time < tmp_location[offset+1,0,-1]:
                        offset = begin
                        break

                    elif bag_time >= tmp_location[offset+1,0,-1]:
                        offset += 1
                        begin += 1

            for i,bag in enumerate(bag_map['loc_bags'][pos_name]):


                if len(bag) > loc_bag_size:
                    divergence_list = []
                    bag_loc = self.location[pos_index][bag]
                    for loc in bag_loc:
                        center = (loc[0,-1] + loc[-1,-1])/2
                        div = np.abs(self.labels[i,-1] - center)
                        divergence_list.append(div)

                    min_indices = np.argpartition(divergence_list,loc_bag_size)[:loc_bag_size]
                    bag_map['loc_bags'][pos_name][i] = [bag[index] for index in min_indices]


        return bag_map

    def select_location(self,user,day,position,start):
        output = []
        found = False

        if user == 3 and day >= 2:
            day -= 1

        for loc_sample in self.location[position][start:]:

            if loc_sample[0,-3] == user and loc_sample[0,-2] == day:

                found = True
                output.append(loc_sample)

            elif found:
                break

        return np.array(output)

    def init_transformers(self):

        use_specto = self.shl_args.train_args['spectograms']
        pos_list = self.shl_args.train_args['positions']
        acc_list = self.shl_args.train_args['acc_signals']
        loc_list = self.shl_args.train_args['loc_signals']
        acc_fusion = self.shl_args.train_args['acc_fusion']
        loc_fusion = self.shl_args.train_args['loc_fusion']
        acc_freq = int(1. / self.shl_args.data_args['smpl_acc_period'])
        acc_bags = self.shl_args.train_args['accBagSize']
        loc_bags = self.shl_args.train_args['locBagSize']


        if use_specto:

            self.accTfrm = SpectogramAccTransform(n_bags=acc_bags,
                                                  pos_name_list=pos_list,
                                                  signal_name_list=acc_list,
                                                  fusion=acc_fusion,
                                                  freq=acc_freq)

            self.accShape = self.accTfrm.get_shape(self.shl_args)

        else:

            self.accTfrm = TemporalAccTransform(n_bags=acc_bags,
                                                pos_name_list=pos_list,
                                                signal_name_list=acc_list,
                                                fusion=acc_fusion)

            self.accShape = self.accTfrm.get_shape(self.shl_args)

        self.locTfrm = TemporalLocationTransform(n_bags=loc_bags,
                                                 pos_name_list=pos_list,
                                                 signals_name_list=loc_list,
                                                 fusion=loc_fusion)

        self.locShape = self.locTfrm.get_shape(self.shl_args)

        self.lbsTfrm = CategoricalTransform()

        if len(self.shl_args.train_args['positions'])==1:

            self.inputShape = (self.accShape,self.locShape)

            self.inputType = (tf.float64,tf.float64)

        else:

            self.inputShape = ({pos: self.accShape for pos in self.shl_args.train_args['positions']},
                                {pos: self.locShape for pos in self.shl_args.train_args['positions']})

            self.inputType = ({pos: tf.float64 for pos in self.shl_args.train_args['positions']},
                                {pos: tf.float64 for pos in self.shl_args.train_args['positions']})


    def to_generator(self,is_test = False):


        positions = ['Torso',
                      'Hips',
                      'Bag',
                      'Hand']



        if not is_test:

            indices = self.dataIndices[:self.trainSize]

        else:

            indices = self.dataIndices[self.trainSize:]


        def gen():

            for i in indices:

                transformedAccBag = self.accTfrm(self.acceleration[self.acc_bags[i]])


                locBag = []
                for pos_i,position in enumerate(positions):

                    locBag.append(self.location[pos_i][self.loc_bags[position][i]])

                transformedLocBag = self.locTfrm(locBag)
                del locBag

                y = self.lbsTfrm(self.labels[self.lbs_bags[i]][0])


                yield (transformedAccBag , transformedLocBag),y

        return tf.data.Dataset.from_generator(
            gen,

            output_types = (self.inputType,
                            tf.float16),

            output_shapes = (self.inputShape,
                             (8))
        )

    def batch_and_prefetch(self,train,test):

        return train.shuffle(1000).repeat()\
                   .batch(batch_size=self.trainBatchSize)\
                   .prefetch(tf.data.AUTOTUNE), \
                test.shuffle(self.testSize).repeat()\
                    .batch(batch_size=self.testBatchSize)\
                    .prefetch(tf.data.AUTOTUNE)


    def shuffle(self):
        self.dataIndices = [*range(self.bags)]
        random.shuffle(self.dataIndices)

        self.trainSize = self.bags - self.shl_args.train_args['test_size']
        self.testSize = self.shl_args.train_args['test_size']
        self.trainBatchSize = self.shl_args.train_args['batchSize']
        self.testBatchSize = self.shl_args.train_args['testBatchSize']

    def __call__(self):

        self.init_transformers()

        bags = self.to_bags()
        self.acc_bags , self.lbs_bags , self.loc_bags = bags['acc_bags'] , bags['lbs_bags'] , bags['loc_bags']
        del bags

        self.shuffle()

        train = self.to_generator(is_test=False)
        test = self.to_generator(is_test=True)


        return self.batch_and_prefetch(train,test)


