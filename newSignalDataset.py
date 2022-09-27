import random

import numpy as np
import pandas as pd

from configParser import Parser
from extractData import extractData
from newPreprocessData import preprocessData
import tensorflow as tf
from newTransformers import *
from sklearn.model_selection import train_test_split
from collections import Counter, OrderedDict
from tqdm import tqdm
from hmmlearn import hmm


np.set_printoptions(precision=14)
class SignalsDataset:

    def __init__(self,
                 regenerate = False,
                 deleteFolders = False,
                 verbose = False
                 ):



        parser = Parser()
        self.shl_args = parser.get_args()
        self.dynamicWindow = self.shl_args.data_args['dynamicWindow']
        self.locSampling = self.shl_args.data_args['locSampling']

        self.verbose = verbose

        if not regenerate:
            exData = extractData(self.shl_args)

            if not exData.found:



                ppData = preprocessData(args=self.shl_args,
                                         verbose=verbose)

                ppData()
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
                                     delete_dst=False,
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
        self.trainBatchSize = self.shl_args.train_args['trainBatchSize']
        self.valBatchSize = self.shl_args.train_args['valBatchSize']
        self.testBatchSize = self.shl_args.train_args['testBatchSize']


        if self.shl_args.train_args['drop_run']:
            self.n_labels = 7

        else:
            self.n_labels = 8

        self.bag_stride = self.shl_args.train_args['bagStride']
        self.padding_method = self.shl_args.train_args['padding_method']

    def to_bags(self):


        pos_names = ['Torso',
                      'Hips',
                      'Bag',
                      'Hand']

        bag_map = {
            'acc_bags' : [[] for _ in range(self.bags)],
            'lbs_bags' : [i for i in range(self.bags)],
            'loc_bags' : {}
        }



        samples = self.acceleration.shape[0]

        acc_bag_size = self.shl_args.train_args['accBagSize']
        loc_bag_size = self.shl_args.train_args['locBagSize']

        contain_label = self.shl_args.train_args['containLabel']

        intersect = self.shl_args.train_args['intersect']

        accPivot = self.shl_args.train_args['accBagPivot']
        locPivot = self.shl_args.train_args['locBagPivot']

        locPos = self.shl_args.data_args['locPosition']

        bagged = self.shl_args.data_args['bagging']


        if not accPivot:
            accPivot = acc_bag_size - 1

        if not locPivot:
            locPivot = loc_bag_size - 1

        if not locPos:
            locPos = self.shl_args.data_args['locDuration'] - 1


        if self.locSampling in ['window','hop']:

            pivot = 0
            i = 0

            while pivot < self.bags:


                label = self.labels[pivot]

                sample_index = label[1]
                label_index = label[2]

                bag_user = label[-3]
                bag_day = label[-2]
                bag_time = label[-1]

                bag_map['acc_bags'][i].append(sample_index)

                if not bagged:

                    if intersect:

                        bag_start_time = self.acceleration[sample_index][0][-1]
                        bag_end_time = self.acceleration[sample_index][-1][-1]

                    step = 1

                    while step <= accPivot:
                        moved_index = sample_index - step
                        if moved_index >= 0:
                            pair_acc = self.acceleration[moved_index][label_index]

                            start_time = self.acceleration[moved_index][0][-1]
                            end_time = self.acceleration[moved_index][-1][-1]

                            if pair_acc[-3] == bag_user and pair_acc[-2] == bag_day:
                                if contain_label:
                                    if start_time <= bag_time <= end_time:
                                        bag_map['acc_bags'][i].insert(0, moved_index)

                                    else:
                                        break

                                elif intersect:

                                    if end_time > bag_start_time and start_time < bag_end_time:
                                        bag_map['acc_bags'][i].insert(0, moved_index)

                                    else:
                                        break

                                else:
                                    bag_map['acc_bags'][i].insert(0, moved_index)

                        step += 1

                    step = 1
                    while step < acc_bag_size - accPivot:
                        moved_index = sample_index + step
                        if moved_index < samples:
                            pair_acc = self.acceleration[moved_index][label_index]

                            start_time = self.acceleration[moved_index][0][-1]
                            end_time = self.acceleration[moved_index][-1][-1]

                            if pair_acc[-3] == bag_user and pair_acc[-2] == bag_day:
                                if contain_label:
                                    if start_time <= bag_time <= end_time:
                                        bag_map['acc_bags'][i].append(moved_index)

                                    else:
                                        break

                                elif intersect:

                                    if end_time > bag_start_time and start_time < bag_end_time:
                                        bag_map['acc_bags'][i].append(moved_index)

                                    else:
                                        break

                                else:
                                    bag_map['acc_bags'][i].append(moved_index)

                        step += 1

                pivot += 1
                i += 1


            if not contain_label and intersect:
                original_bag_size = loc_bag_size
                loc_bag_size = 1

            for pos_index, pos_name in enumerate(pos_names):

                offset = 0
                last = False
                bag_map['loc_bags'][pos_name] = [[] for _ in range(self.bags)]
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

                            if self.dynamicWindow:
                                filled = np.sum(np.count_nonzero(tmp_location[offset] == -1, axis=1) == 0)


                                if filled < self.shl_args.train_args['padding_threshold']:

                                    pass

                                elif self.shl_args.train_args['pair_threshold']:
                                    distance = np.abs(tmp_location[offset, locPos, -1] - bag_time)
                                    if distance > self.shl_args.train_args['pair_threshold']:

                                        pass

                                    else:

                                        bag_map['loc_bags'][pos_name][i].append(offset + start)

                                else:
                                    bag_map['loc_bags'][pos_name][i].append(offset + start)

                            else:
                                bag_map['loc_bags'][pos_name][i].append(offset + start)
                            offset += 1


                        elif offset == tmp_n-1 or \
                        tmp_location[offset,-1,-1] < bag_time < tmp_location[offset+1,0,-1]:
                            offset = begin
                            break


                        elif bag_time >= tmp_location[offset+1, 0, -1]:
                            offset += 1
                            begin += 1


                tmp_bag_map = []

                for i,bag in enumerate(bag_map['loc_bags'][pos_name]):

                    if len(bag) > loc_bag_size:
                        divergence_list = []
                        bag_loc = self.location[pos_index][bag]
                        for loc in bag_loc:

                            timestamp = loc[locPos, -1]
                            div = np.abs(self.labels[i,-1] - timestamp)
                            divergence_list.append(div)

                        min_indices = np.argpartition(divergence_list,loc_bag_size)[:loc_bag_size]
                        tmp_bag_map.append([bag[index] for index in min_indices])


                    elif len(bag) <= loc_bag_size:
                        tmp_bag_map.append(bag)



                bag_map['loc_bags'][pos_name] = tmp_bag_map



                if not contain_label and intersect and original_bag_size>1:

                    for map_index, bag in enumerate(bag_map['loc_bags'][pos_name]):

                        try:
                            loc_index = bag[0]

                        except:
                            continue


                        bag_user = self.location[pos_index][loc_index][0][-3]
                        bag_day = self.location[pos_index][loc_index][0][-2]
                        bag_start_time = self.location[pos_index][loc_index][0][-1]
                        bag_end_time = self.location[pos_index][loc_index][-1][-1]


                        step = 1
                        while step <= locPivot:
                            moved_index = loc_index - step
                            if moved_index >= 0:
                                start_time = self.location[pos_index][moved_index][0][-1]
                                end_time = self.location[pos_index][moved_index][-1][-1]
                                user = self.location[pos_index][moved_index][0][-3]
                                day = self.location[pos_index][moved_index][0][-2]

                                if user == bag_user and day == bag_day:
                                    if end_time > bag_start_time and start_time < bag_end_time:
                                        bag_map['loc_bags'][pos_name][map_index].insert(0, moved_index)

                            step += 1


                        step = 1
                        while step < loc_bag_size - locPivot:
                            moved_index = loc_index + step
                            if moved_index < self.location[pos_index].shape[0]:
                                start_time = self.location[pos_index][moved_index][0][-1]
                                end_time = self.location[pos_index][moved_index][-1][-1]
                                user = self.location[pos_index][moved_index][0][-3]
                                day = self.location[pos_index][moved_index][0][-2]

                                if user == bag_user and day == bag_day:
                                    if end_time > bag_start_time and start_time < bag_end_time:
                                        bag_map['loc_bags'][pos_name][map_index].append(moved_index)

                            step += 1

            return bag_map

        elif self.locSampling == 'labelBased':

            for pos_index, pos_name in enumerate(pos_names):
                bag_map['loc_bags'][pos_name] = [[] for _ in range(self.bags)]

            pivot = 0
            i = 0

            while pivot < self.bags:

                label = self.labels[pivot]

                sample_index = label[1]
                label_index = label[2]
                bag_user = label[-3]
                bag_day = label[-2]
                bag_time = label[-1]


                for pos_index, pos_name in enumerate(pos_names):

                    location_index = np.argwhere(self.location[pos_index][:, 0, -2] == sample_index)
                    if np.size(location_index):
                        bag_map['loc_bags'][pos_name][i].append(int(location_index))



                bag_map['acc_bags'][i].append(sample_index)


                if intersect:
                    bag_start_time = self.acceleration[sample_index][0][-1]
                    bag_end_time = self.acceleration[sample_index][-1][-1]


                step = 1

                while step <= accPivot:
                    moved_index = sample_index - step

                    if 0 <= moved_index:
                        pair_acc = self.acceleration[moved_index][label_index]

                        start_time = self.acceleration[moved_index][0][-1]
                        end_time = self.acceleration[moved_index][-1][-1]

                        if pair_acc[-3] == bag_user and pair_acc[-2] == bag_day:
                            if contain_label:
                                if start_time <= bag_time <= end_time:
                                    bag_map['acc_bags'][i].insert(0, moved_index)

                                else:
                                    break

                            elif intersect:

                                if end_time > bag_start_time and start_time < bag_end_time:
                                    bag_map['acc_bags'][i].insert(0, moved_index)

                                else:
                                    break

                            else:
                                bag_map['acc_bags'][i].insert(0, moved_index)

                    step += 1



                step = 1

                while step < acc_bag_size - accPivot:
                    moved_index = sample_index + step

                    if moved_index < samples:
                        pair_acc = self.acceleration[moved_index][label_index]

                        start_time = self.acceleration[moved_index][0][-1]
                        end_time = self.acceleration[moved_index][-1][-1]

                        if pair_acc[-3] == bag_user and pair_acc[-2] == bag_day:
                            if contain_label:
                                if start_time <= bag_time <= end_time:
                                    bag_map['acc_bags'][i].append(moved_index)

                                else:
                                    break

                            elif intersect:

                                if end_time > bag_start_time and start_time < bag_end_time:
                                    bag_map['acc_bags'][i].append(moved_index)

                                else:
                                    break

                            else:
                                bag_map['acc_bags'][i].append(moved_index)

                    step += 1

                pivot += 1
                i += 1

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

    def init_transformers(self, baseline = False, simCLR = False, accTransfer = False, locTransfer = False):

        use_specto = self.shl_args.train_args['spectograms']
        use_fft = self.shl_args.train_args['FFT']

        if not locTransfer:
            if use_specto:

                self.accTfrm = SpectogramAccTransform(
                    self.shl_args,
                    baseline=baseline,
                    simCLR=simCLR
                )


            elif use_fft:

                self.accTfrm = FastFourierTransform(
                    self.shl_args,
                    baseline = baseline,
                    simCLR = simCLR
                )


            else:

                self.accTfrm = TemporalAccTransform(
                    shl_args=self.shl_args,
                    baseline=baseline,
                    simCLR=simCLR
                )

            self.accShape = self.accTfrm.get_shape()



        if not accTransfer:
            self.locTfrm = TemporalLocationTransform(shl_args=self.shl_args, baseline=baseline)
            self.locSignalsShape, self.locFeaturesShape = self.locTfrm.get_shape()


        self.lbsTfrm = CategoricalTransform(self.shl_args.train_args['drop_run'])



        if not simCLR:

            if locTransfer:
                self.inputShape = (self.locSignalsShape, self.locFeaturesShape)

                self.inputType = (tf.float64, tf.float64)

            elif accTransfer:
                self.inputShape = self.accShape

                self.inputType = tf.float64

            else:

                self.inputShape = (self.accShape, self.locSignalsShape, self.locFeaturesShape)

                self.inputType = (tf.float64, tf.float64, tf.float64)

        else:

            self.inputShape = (self.accShape, self.accShape)

            self.inputType = (tf.float64, tf.float64)

    def accFeatures(self,acceleration,position):


        positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }


        pos_i = 3 * positions[position]

        magnitude = np.sqrt(np.sum(acceleration[:, :, pos_i:pos_i + 3] ** 2,
                                axis=2))[0]


        var = np.var(magnitude)

        freq_acc = np.fft.fft(magnitude)
        freq_magnitude = np.power(np.abs(freq_acc),2)

        coef1Hz = freq_magnitude[1]
        coef2Hz = freq_magnitude[2]
        coef3Hz = freq_magnitude[3]

        acc_features = [var, coef1Hz, coef2Hz, coef3Hz]

        return acc_features

    def calc_haversine_dis(self, lat, lon, alt, moment):

        point1 = (lat[moment-1],lon[moment-1])
        point2 = (lat[moment],lon[moment])
        return math.sqrt(great_circle(point1,point2).m ** 2 + (alt[moment] - alt[moment-1]) ** 2)

    def calc_haversine_vel(self, lat, lon, alt, t, moment):
        hvs_dis = self.calc_haversine_dis(lat,lon,alt,moment)
        return 1000. * hvs_dis / (t[moment] - t[moment-1])

    def haversine_velocity(self, pos_location, duration):
        time_signal = pos_location[:, -1]
        x_signal = pos_location[:, 1]
        y_signal = pos_location[:, 2]
        z_signal = pos_location[:, 3]


        vel_signal = np.zeros((duration - 1))


        for moment in range(1, duration):
            vel_signal[moment - 1] = self.calc_haversine_vel(x_signal,
                                                             y_signal,
                                                             z_signal,
                                                             time_signal,
                                                             moment)

        return vel_signal

    def locFeatures(self,location):

        positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }
        pos_name = self.shl_args.train_args['gpsPosition']


        pos_location = location[positions[pos_name]]

        if np.size(pos_location):
            pos_location = pos_location[0]

        else:
            return -1


        for location_timestamp in pos_location:
            if location_timestamp[0] == -1.:
                return -1


        velocity = self.haversine_velocity(pos_location, self.shl_args.data_args['locDuration'])
        # print(pos_location)
        # print(velocity)


        return velocity[0]




        

    def to_pandas(self, is_val = False, is_test = False, motorized_class = True, DHMM = False):
        positions = ['Torso',
                      'Hips',
                      'Bag',
                      'Hand']


        acc_positions = self.shl_args.train_args['positions']


        if not is_test:

            if not is_val:
                indices = self.train_indices

            else:

                indices = self.val_indices

        else:

            indices = self.test_indices

        data = []
        labels = []
        for en, index in enumerate(indices):

            i = index[0]
            position = acc_positions[index[1]]

            # print(i)
            # print(position)

            locBag = []
            for pos_i, pos in enumerate(positions):
                locBag.append(copy.deepcopy(self.location[pos_i][self.loc_bags[pos][i]]))

            LocFeature = self.locFeatures(locBag)



            if LocFeature == -1:

                continue

            del locBag



            AccFeatures = self.accFeatures(self.acceleration[self.acc_bags[i]],
                                             position=position)

            # print(self.labels[self.lbs_bags[i]])

            Lb = self.labels[self.lbs_bags[i]][0]-1
            if motorized_class:
                Lb = Lb if Lb<4 else 4

            Time = self.labels[self.lbs_bags[i]][-1]


            if en == 0:
                data = [[LocFeature, *AccFeatures]]
                labels = [Lb]
                time = [Time]

            else:
                data.append([LocFeature, *AccFeatures])
                labels.append(Lb)
                time.append(Time)



        if DHMM:
            df_data = [pd.DataFrame(data[i::4], columns=['vel','acc_var','acc_DFT_1Hz','acc_DFT_2Hz','acc_DFT_3Hz'], dtype=float) for i in range(4)]

            df_labels = [pd.DataFrame(labels[i::4], columns=['label'], dtype=int) for i in range(4)]

            time = pd.DataFrame(time[0::4], columns=['time'], dtype=float)


            dT_threshold = 2000
            time['dT'] = time['time'].diff().abs()
            split = time.index[time['dT'] > dT_threshold].tolist()


            last_check = 0
            split_data = []
            split_lbs = []
            for index in split:
                split_data.extend([df_data[i].loc[last_check:index - 1] for i in range(4)])
                split_lbs.extend([df_labels[i].loc[last_check:index - 1] for i in range(4)])
                last_check = index


            if not is_test:
                classes = [i for i in range(5)] if motorized_class else [i for i in range(self.n_labels)]

                transition_mx = None
                for i,seq in enumerate(split_lbs):
                    seq_ = seq
                    seq_['label_'] = seq_.shift(-1)

                    groups = seq_.groupby(['label','label_'])
                    counts = {i[0]:len(i[1]) for i in groups}



                    matrix = pd.DataFrame()

                    for x in classes:
                        matrix[x] = pd.Series([counts.get((x, y), 0) for y in classes], index=classes)

                    if i!=0:
                        transition_mx = transition_mx.add(matrix)

                    else:
                        transition_mx = matrix



                transition_mx["sum"] = transition_mx.sum(axis=1)
                transition_mx = transition_mx.div(transition_mx["sum"], axis=0)

                return split_data, split_lbs, transition_mx

            return split_data, split_lbs

        else:
            data = pd.DataFrame(data, columns=['vel', 'acc_var', 'acc_DFT_1Hz', 'acc_DFT_2Hz', 'acc_DFT_3Hz'],
                                dtype=float)

            labels = pd.DataFrame(labels, columns=['label'], dtype=int)
            time = pd.DataFrame(time, columns=['time'], dtype=float)



            return data, labels

    def to_generator(self, is_val = False, is_test = False, simCLR = False, accTransfer = False, locTransfer = False, criterion = None):


        positions = ['Torso',
                      'Hips',
                      'Bag',
                      'Hand']

        acc_positions = self.shl_args.train_args['positions']
        bagged = self.shl_args.data_args['bagging']
        size = self.shl_args.data_args['accBagSize'] if bagged else None
        stride = self.shl_args.data_args['accBagStride'] if bagged else None
        duration = self.shl_args.data_args['accDuration'] if bagged else None

        if not is_test:
            if not is_val:
                indices = self.train_indices

            else:

                indices = self.val_indices

        else:

            indices = self.test_indices


        def gen():

            for index in indices:

                if simCLR:
                    if criterion == 'position':
                        i = index[0]
                        position1 = acc_positions[index[1][0]]
                        position2 = acc_positions[index[1][1]]

                    elif criterion == 'augmentation':
                        i = index[0]
                        position = acc_positions[index[1]]

                elif not locTransfer:
                    i = index[0]
                    position = acc_positions[index[1]]

                else:
                    i = index
                    position = None


                if simCLR:
                    if criterion == 'position':
                        transformedAccBag1 = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[i]]),
                                                         is_train = not (is_val or is_test),
                                                         position = position1)

                        transformedAccBag2 = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[i]]),
                                                         is_train = not (is_val or is_test),
                                                         position = position2)

                    elif criterion == 'augmentation':
                        transformedAccBag1 = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[i]]),
                                                          is_train= True,
                                                          position=position)

                        transformedAccBag2 = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[i]]),
                                                          is_train= True,
                                                          position=position)


                elif not locTransfer:



                    transformedAccBag = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[i]])
                                                     , is_train = not (is_val or is_test),
                                                     position = position, bagged = bagged,
                                                     size = size, stride = stride,
                                                     acc = accTransfer)



                if not simCLR:
                    if not accTransfer:
                        locBag = []
                        for pos_i,position in enumerate(positions):

                            locBag.append(copy.deepcopy(self.location[pos_i][self.loc_bags[position][i]]))

                        transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(locBag,
                                                         is_train = not (is_val or is_test))



                        del locBag

                    y = self.lbsTfrm(self.labels[self.lbs_bags[i]][0])

                    if locTransfer:
                        yield (transformedLocSignalsBag, transformedLocFeaturesBag),y

                    elif accTransfer:
                        yield transformedAccBag, y

                    else:
                        yield (transformedAccBag, transformedLocSignalsBag, transformedLocFeaturesBag), y

                else:

                    yield transformedAccBag1, transformedAccBag2

        if not simCLR:
            return tf.data.Dataset.from_generator(
                gen,

                output_types = (self.inputType,
                                tf.float32),

                output_shapes = (self.inputShape,
                                 (self.n_labels))
            )

        else:
            return tf.data.Dataset.from_generator(
                gen,

                output_types=self.inputType,
                output_shapes=self.inputShape
            )

    def batch_and_prefetch(self, train, val = None, test = None):

        if val and test:
            if self.dynamicWindow and self.padding_method == 'variableLength':

                return train.cache().repeat()\
                             .batch(batch_size=self.trainBatchSize).\
                             shuffle(500)\
                            .prefetch(tf.data.AUTOTUNE), \
                        val.batch(batch_size=self.valBatchSize).prefetch(tf.data.AUTOTUNE),\
                        test.batch(batch_size=self.testBatchSize).prefetch(tf.data.AUTOTUNE)


            else:
                return train.cache().shuffle(1000).repeat()\
                            .batch(batch_size=self.trainBatchSize)\
                            .prefetch(tf.data.AUTOTUNE), \
                        val.cache().shuffle(1000).repeat()\
                            .batch(batch_size=self.valBatchSize).prefetch(tf.data.AUTOTUNE),\
                        test.cache().shuffle(1000).repeat()\
                           .batch(batch_size=self.testBatchSize).prefetch(tf.data.AUTOTUNE)

        else:
            if self.dynamicWindow and self.padding_method == 'variableLength':

                return train.cache().repeat() \
                           .batch(batch_size=self.trainBatchSize) \
                            .shuffle(500)\
                           .prefetch(tf.data.AUTOTUNE)


            else:
                return train.cache().shuffle(1000).repeat() \
                           .batch(batch_size=self.trainBatchSize) \
                           .prefetch(tf.data.AUTOTUNE)


    def to_batches(self):

        pos_name_list = self.shl_args.train_args['positions']


        if pos_name_list == None:
            pos_name_list = ['Hand']


        positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }

        min_length = self.shl_args.train_args['padding_threshold']
        pos_name = pos_name_list[0]
        pos_i = positions[pos_name]


        seq_lengths = []


        for train_index in self.train_indices:
            pair = self.loc_bags[pos_name][train_index]
            if pair:

                var_length = np.sum(np.count_nonzero(self.location[pos_i][pair][0] == -1, axis=1) == 0)


                if var_length < min_length:
                    seq_lengths.append(0)

                else:
                    seq_lengths.append(var_length)

            else:
                seq_lengths.append(0)

        split_train = [train_index for _ ,train_index in sorted(zip(
            seq_lengths,
            self.train_indices
        ))]


        length_counts = OrderedDict( sorted( Counter(seq_lengths).items()))

        batch = self.shl_args.train_args['trainBatchSize']


        totalCount = 0
        for length, count in length_counts.items():
            totalCount += count

            del split_train[totalCount - count % batch : totalCount]
            totalCount -= totalCount % batch

        split_train = np.reshape(split_train, (-1,batch))
        batches = split_train.shape[0]

        np.random.shuffle(split_train)
        split_train = np.reshape(split_train, (-1)).tolist()

        self.train_indices = split_train

        return batches





    def split_train_val(self, dataIndices, dataSize):

        if self.shl_args.train_args['stratify'] == 'hop':

            valSize = self.shl_args.train_args['val_size']
            stride = dataSize//valSize


            self.val_indices = [dataIndices.pop(i-shift) for shift,i in enumerate([*range(0,dataSize,stride)])]
            random.shuffle(self.val_indices)
            self.valSize = len(self.val_indices)

            if self.valBatchSize == None:
                self.valBatchSize = self.valSize

            self.train_indices = dataIndices[::self.bag_stride]
            random.shuffle(self.train_indices)
            self.trainSize = len(self.train_indices)


        elif self.shl_args.train_args['stratify'] == 'concentrated':


            val_percentage = self.shl_args.train_args['val_percentage']

            originalIndices = dataIndices
            dataIndices = pd.DataFrame(dataIndices, columns = ['index', 'user_label'])

            count = dataIndices['user_label'].value_counts()


            val_count = count*val_percentage
            val_count = val_count.astype('int32')


            val_indices = []
            for user_label, count in val_count.items():

                candidates = pd.DataFrame()
                tmp_count = count

                while candidates.empty:

                    candidates = dataIndices[dataIndices['user_label']==user_label].user_label.groupby([dataIndices.user_label, dataIndices.user_label.diff().ne(0).cumsum()]).transform('size').ge(tmp_count).astype(int)
                    candidates = pd.DataFrame(candidates)
                    candidates = candidates[candidates['user_label'] == 1]
                    tmp_count = int(tmp_count * 0.95)

                index = candidates.sample(random_state=1).index[0]

                val_indices.append(index)

                n_indices = 1
                up = 1
                down = 1
                length = dataIndices.shape[0]

                while n_indices < tmp_count - 1:

                    if index+up<length and user_label == dataIndices.iloc[index + up]['user_label']:
                        val_indices.append(index + up)
                        up += 1
                        n_indices += 1


                    if index - down >= 0 and user_label == dataIndices.iloc[index - down]['user_label']:
                        val_indices.append(index - down)
                        down += 1
                        n_indices += 1

            val_indices.sort()
            self.val_indices = [originalIndices.pop(i - shift)[0] for shift, i in enumerate(val_indices)]

            self.valSize = len(self.val_indices)

            if self.valBatchSize == None:
                self.valBatchSize = self.valSize

            self.train_indices = [x[0] for x in originalIndices[::self.bag_stride]]
            self.trainSize = len(self.train_indices)


            # random.shuffle(self.val_indices)
            # random.shuffle(self.train_indices)

        else:
            self.valSize = self.shl_args.train_args['val_size']
            if self.valBatchSize == None:
                self.valBatchSize = self.valSize


            self.trainSize = dataSize - self.valSize

            random.shuffle(dataIndices)

            dataIndices = pd.DataFrame(dataIndices,columns=['index','user_label'])


            train_indices, val_indices = train_test_split(dataIndices,
                                                            test_size = self.valSize / dataSize,
                                                            stratify = dataIndices['user_label'],
                                                            random_state=22)


            self.train_indices = train_indices['index'].tolist()
            self.train_indices = self.train_indices[::self.bag_stride]
            self.val_indices = val_indices['index'].tolist()


    def split_train_val_test(self, user_seperated = False, firstRound = True, DHMM = False):

        test_user = self.shl_args.train_args['test_user']
        self.test_indices = []
        self.testSize = 0

        if DHMM:
            self.train_indices = []
            self.trainSize = 0

        if user_seperated:
            self.train_indices = []
            self.trainSize = 0

            self.val_indices = []
            self.valSize = 0

            if firstRound:
                if test_user == 1:
                    self.train_user = 2
                    self.val_user = 3

                elif test_user == 2:
                    self.train_user = 1
                    self.val_user = 3

                else:
                    self.train_user = 1
                    self.val_user = 2

            else:
                if test_user == 1:
                    self.train_user = 3
                    self.val_user = 2

                elif test_user == 2:
                    self.train_user = 3
                    self.val_user = 1

                else:
                    self.train_user = 2
                    self.val_user = 1

        train_val_indices = []
        train_val_size = 0

        for index,(label,user) in enumerate(zip(self.labels[:,0],
                                                self.labels[:,-3])):

            if index not in self.run_indices:
                if user == test_user:
                    self.testSize += 1
                    self.test_indices.append(index)

                elif DHMM:
                    self.trainSize += 1
                    self.train_indices.append(index)

                elif user_seperated:
                    if user == self.train_user:
                        self.trainSize += 1
                        self.train_indices.append(index)

                    elif user == self.val_user:
                        self.valSize += 1
                        self.val_indices.append(index)



                else:
                    train_val_size += 1

                    if self.shl_args.train_args['stratify'] == 'user&label':
                        train_val_indices.append([index, str(user) + '_' + str(label)])

                    elif self.shl_args.train_args['stratify'] == 'label':
                        train_val_indices.append([index, str(label)])

                    elif self.shl_args.train_args['stratify'] == 'hop':
                        train_val_indices.append(index)

                    elif self.shl_args.train_args['stratify'] == 'concentrated':
                        train_val_indices.append([index, user * 10 + label])

        # random.shuffle(self.test_indices)

        if not user_seperated and not DHMM:
            self.split_train_val(train_val_indices, train_val_size)

    def to_seq_generator(self,pr_seqs,tr_seqs):
        def gen():

            for pr_seq,tr_seq in zip(pr_seqs,tr_seqs):
                x_forward = [pos_seq[0] for pos_seq in pr_seq]
                x_backward = [pos_seq[1] for pos_seq in pr_seq]
                y = [tr_seq for _ in range(4)]
                yield (x_forward, x_backward), y

        return tf.data.Dataset.from_generator(
            gen,
            output_types=((tf.float32, tf.float32), tf.float32),
            output_shapes=(((4, None, self.n_labels), (4, None, self.n_labels)), (4, self.n_labels))
        )

    def seqs_batch_and_prefetch(self, seqs):
        return seqs.cache().repeat().prefetch(tf.data.AUTOTUNE)



    def get_seq_lbs(self, Model, type = 'LSTM'):

        if type == 'LSTM':

            test_user = self.shl_args.train_args['test_user']

            positions = ['Torso',
                          'Hips',
                          'Bag',
                          'Hand']

            acc_positions = self.shl_args.train_args['positions']

            bagged = self.shl_args.data_args['bagging']
            size = self.shl_args.data_args['accBagSize'] if bagged else None
            stride = self.shl_args.data_args['accBagStride'] if bagged else None


            trans_threshold = self.shl_args.train_args['transition_threshold']


            train_predicted_sequences = []
            train_true_sequences = []
            train_new_tr_sequences = []

            val_predicted_sequences = []
            val_true_sequences = []

            test_predicted_sequences = []
            test_true_sequences = []
            test_new_tr_sequences = []

            num = 0
            train_length, test_length = 0, 0

            train_inputs = [[[] for _ in range(3)] for _ in range(len(acc_positions))]
            test_inputs = [[[] for _ in range(3)] for _ in range(len(acc_positions))]

            for index, (label, day, time, user) in enumerate(zip(self.labels[:-1, 0],
                                                      self.labels[:-1,-2],
                                                      self.labels[:-1,-1],
                                                      self.labels[:-1,-3])):

                if index not in self.run_indices:

                    if user in [self.val_user,test_user]:

                        for pos_j,position in enumerate(acc_positions):

                            transformedAccSignalsBag = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[index]])
                                                             ,is_train=False,
                                                             position=position, bagged=bagged,
                                                             size=size, stride=stride)

                            locBag = []
                            for pos_i, position in enumerate(positions):
                                locBag.append(copy.deepcopy(self.location[pos_i][self.loc_bags[position][index]]))


                            transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(locBag, is_train=False)

                            if user == self.val_user:
                                if pos_j == 0:
                                    train_length += 1
                                    train_new_tr_sequences.append(self.lbsTfrm(label))
                                train_inputs[pos_j][0].append(transformedAccSignalsBag)
                                train_inputs[pos_j][1].append(transformedLocSignalsBag)
                                train_inputs[pos_j][2].append(transformedLocFeaturesBag)

                            elif user == test_user:
                                if pos_j == 0:
                                    test_length += 1
                                    test_new_tr_sequences.append(self.lbsTfrm(label))
                                test_inputs[pos_j][0].append(transformedAccSignalsBag)
                                test_inputs[pos_j][1].append(transformedLocSignalsBag)
                                test_inputs[pos_j][2].append(transformedLocFeaturesBag)


                        if self.labels[index + 1][-1] - time > trans_threshold \
                                or self.labels[index + 1][-2] != day or self.labels[index + 1][-3] != user:



                            if user == self.val_user:
                                train_inputs_ = [[np.array(train_inputs[pos_j][i]) for i in range(3)] for pos_j in range(len(acc_positions))]

                                num += 1

                                train_new_pd_sequences = [np.zeros((train_length,self.n_labels)) for _ in
                                                          range(len(acc_positions))]

                                top_indices = [np.argmax(Model.call(train_inputs_[pos_j]),axis=1) for pos_j in
                                                          range(len(acc_positions))]

                                for pos_j in range(len(acc_positions)):
                                    train_new_pd_sequences[pos_j][[i for i in range(train_length)],top_indices[pos_j]] = 1.


                                if num % 3==0:


                                    val_predicted_sequences.extend([[[pd_sequence[:j+1], pd_sequence[j:][::-1]] for
                                                                     pd_sequence in train_new_pd_sequences] for j in
                                                                    range(train_length)])
                                    val_true_sequences.extend([train_new_tr_sequences[j] for j in range(train_length)])


                                else:

                                    train_predicted_sequences.extend([[[pd_sequence[:j+1], pd_sequence[j:][::-1]] for
                                                                       pd_sequence in train_new_pd_sequences] for j in
                                                                      range(train_length)])
                                    train_true_sequences.extend([train_new_tr_sequences[j] for j in range(train_length)])

                                train_inputs = [[[] for _ in range(3)] for _ in range(len(acc_positions))]
                                train_new_tr_sequences = []
                                train_length = 0


                            elif user == test_user:



                                test_inputs_ = [[np.array(test_inputs[pos_j][i]) for i in range(3)] for pos_j in range(len(acc_positions))]


                                test_new_pd_sequences = [np.zeros((test_length,self.n_labels)) for _ in
                                                          range(len(acc_positions))]

                                top_indices = [np.argmax(Model.call(test_inputs_[pos_j]),axis=1) for pos_j in
                                                          range(len(acc_positions))]

                                for pos_j in range(len(acc_positions)):
                                    test_new_pd_sequences[pos_j][[i for i in range(test_length)],top_indices[pos_j]] = 1.



                                test_predicted_sequences.extend([[[pd_sequence[:j + 1], pd_sequence[j:][::-1]] for
                                                                 pd_sequence in test_new_pd_sequences] for j in
                                                                range(test_length)])
                                test_true_sequences.extend([test_new_tr_sequences[j] for j in range(test_length)])



                                test_inputs = [[[] for _ in range(3)] for _ in range(len(acc_positions))]
                                test_new_tr_sequences = []
                                test_length = 0

            return train_predicted_sequences, train_true_sequences, val_predicted_sequences, val_true_sequences, test_predicted_sequences, test_true_sequences


        if type == 'Gaussian':
            positions = ['Torso',
                          'Hips',
                          'Bag',
                          'Hand']

            acc_positions = self.shl_args.train_args['positions']

            bagged = self.shl_args.data_args['bagging']
            size = self.shl_args.data_args['accBagSize'] if bagged else None
            stride = self.shl_args.data_args['accBagStride'] if bagged else None

            test_user = self.shl_args.train_args['test_user']
            trans_threshold = self.shl_args.train_args['transition_threshold']
            self.transmat_forward = np.array([[0 for _ in range(self.n_labels)] for _ in range(self.n_labels)])
            self.transmat_backward = np.array([[0 for _ in range(self.n_labels)] for _ in range(self.n_labels)])

            pred_probs = [[] for _ in range(self.n_labels)]
            val_indices = np.array(self.val_indices)

            for index, (label, day, time, user) in enumerate(zip(self.labels[:-1, 0],
                                                      self.labels[:-1,-2],
                                                      self.labels[:-1,-1],
                                                      self.labels[:-1, -3])):

                if index not in self.run_indices:
                    if user != test_user:

                        if index in val_indices[:,0]:
                            for pos_j,position in enumerate(acc_positions):


                                transformedAccBag = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[index]])
                                                                 , is_train=False,
                                                                 position=position, bagged=bagged,
                                                                 size=size, stride=stride)

                                locBag = []
                                for pos_i, position in enumerate(positions):
                                    locBag.append(copy.deepcopy(self.location[pos_i][self.loc_bags[position][index]]))


                                transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(locBag, is_train=False)


                                y_predicted = Model.call([np.array([transformedAccBag]), np.array([transformedLocSignalsBag]), np.array([transformedLocFeaturesBag])])


                                pred_probs[label-1].append(y_predicted[0].numpy().tolist())



                        if self.labels[index + 1][-1] - time < trans_threshold \
                                and self.labels[index + 1][-2] == day\
                                and self.labels[index + 1][-3] == user:
                            self.transmat_forward[label-1][self.labels[index + 1][0]-1] += 1
                            self.transmat_backward[self.labels[index + 1][0]-1][label-1] += 1

            means = np.zeros((8,8))
            covars = np.zeros((8,8,8))

            for k,lb_probs in enumerate(pred_probs):
                means[k] = np.mean(np.transpose(lb_probs),axis=1)
                covars[k] = np.cov(lb_probs,rowvar=False)


            print(means)
            print(covars)


            transmat_forward = self.transmat_forward / self.transmat_forward.sum(axis=1)[:, np.newaxis]


            hmmModel = hmm.GaussianHMM(n_components=self.n_labels, covariance_type='full')

            hmmModel.startprob_ = [0.125 for _ in range(self.n_labels)]
            hmmModel.transmat_ = transmat_forward
            hmmModel.means_ = means
            hmmModel.covars_ = covars





            predicted_sequences = []
            new_pd_sequences = [[] for _ in range(len(acc_positions))]
            true_sequences = []
            new_tr_sequences = [[] for _ in range(len(acc_positions))]

            for index, (label, day, time, user) in enumerate(zip(self.labels[:-1, 0],
                                                      self.labels[:-1,-2],
                                                      self.labels[:-1,-1],
                                                      self.labels[:-1, -3])):

                if index not in self.run_indices:
                    if user == test_user:

                        for pos_j,position in enumerate(acc_positions):

                            transformedAccBag = self.accTfrm(copy.deepcopy(self.acceleration[self.acc_bags[index]])
                                                             , is_train=False,
                                                             position=position, bagged=bagged,
                                                             size=size, stride=stride)

                            locBag = []
                            for pos_i, position in enumerate(positions):
                                locBag.append(copy.deepcopy(self.location[pos_i][self.loc_bags[position][index]]))


                            transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(locBag, is_train=False)


                            y_predicted = Model.call([np.array([transformedAccBag]), np.array([transformedLocSignalsBag]), np.array([transformedLocFeaturesBag])])


                            new_pd_sequences[pos_j].append(y_predicted[0].numpy().tolist())
                            new_tr_sequences[pos_j].append(label - 1)


                        if self.labels[index + 1][-1] - time > trans_threshold \
                                or self.labels[index + 1][-2] != day:

                            for pd_sequence, tr_sequence in zip(new_pd_sequences,new_tr_sequences):


                                q, hmm_sequence = hmmModel.decode(pd_sequence)

                                print(np.argmax(pd_sequence,axis=1))
                                print(hmm_sequence)
                                print(np.array(tr_sequence))
                                print(q)
                                print()



                            new_pd_sequences = [[] for _ in range(len(acc_positions))]
                            new_tr_sequences = [[] for _ in range(len(acc_positions))]



            #
            # train_sequences = []
            # for l in range(n_seqs//2):
            #     train_sequences.extend(predicted_sequences[l])
            #
            # print(np.argmax(train_sequences,axis=1))
            # train_lengths = lengths[:n_seqs // 2]
            #
            # test_sequences = []
            # for l in range(n_seqs//2,len(predicted_sequences)):
            #     test_sequences.extend(predicted_sequences[l])
            #
            # print(np.argmax(test_sequences, axis=1))
            # test_lengths = lengths[n_seqs // 2:]
            #
            # n_predicted_sequences = []
            # for l in range(len(predicted_sequences)):
            #     n_predicted_sequences.extend(predicted_sequences[l])
            # predicted_sequences = n_predicted_sequences
            #
            # models = list()
            # scores = list()
            #
            # for r_s in range(30):
            #     hmmModel = hmm.GMMHMM(n_components=self.n_labels, covariance_type='full', n_iter=100, random_state=r_s)
            #     hmmModel.fit(train_sequences, lengths = train_lengths)
            #     models.append(hmmModel)
            #     scores.append(hmmModel.score(test_sequences, lengths = test_lengths))
            #     print(scores)
            #
            # hmmModel = models[np.argmax(scores)]
            # hmm_sequences = hmmModel.predict(predicted_sequences, lengths = lengths)
            # for pd_sequence, tr_sequence, hmm_sequence in zip(predicted_sequences,true_sequences, hmm_sequences):
            #     print(pd_sequence)
            #     print(np.argmax(pd_sequence),tr_sequence,hmm_sequence)
            #

            return predicted_sequences, true_sequences

    def drop_run(self):

        for index,label in enumerate(self.labels[:,0]):
            if label == 3:
                self.run_indices.append(index)


    def get_simCLR_bags(self, criterion):



        positions = len(self.shl_args.train_args['positions'])

        test_user = self.shl_args.train_args['test_user']
        val_indices = []
        self.test_indices = []


        for index,user in enumerate(self.acceleration[:,0,-3]):
            if user == test_user:
                self.test_indices.append(index)


        for index in self.val_indices:
            val_indices.append(self.acc_bags[index][0])

        self.val_indices = val_indices


        acc_indices = [*range(self.acceleration.shape[0])]
        simCLR_bags = [[index] for index in acc_indices]

        self.train_indices = [index for index in acc_indices
                              if index not in self.test_indices and index not in self.val_indices]

        if criterion == 'augmentation':
            for i, test_index in enumerate(self.test_indices):
                if i == 0:

                    pos_test_indices = [[test_index, pos] for pos in range(positions)]

                    continue


                pos_test_indices.extend([[test_index, pos] for pos in range(positions)])

            self.test_indices = pos_test_indices

            random.shuffle(self.test_indices)
            self.testSize = len(self.test_indices)

            for i, val_index in enumerate(self.val_indices):
                if i == 0:

                    pos_val_indices = [[val_index, pos] for pos in range(positions)]
                    continue


                pos_val_indices.extend([[val_index, pos] for pos in range(positions)])

            self.val_indices = pos_val_indices

            random.shuffle(self.val_indices)
            self.valSize = len(self.val_indices)

            for i, train_index in enumerate(self.train_indices):
                if i == 0:

                    pos_train_indices = [[train_index, pos] for pos in range(positions)]
                    continue

                pos_train_indices.extend([[train_index, pos] for pos in range(positions)])

            self.train_indices = pos_train_indices

            random.shuffle(self.train_indices)
            self.trainSize = len(self.train_indices)

        if criterion == 'position':
            for i, test_index in enumerate(self.test_indices):
                if i == 0:
                    random_positions = random.sample([*range(positions)],2*(positions//2))
                    pos_test_indices = [[test_index, [random_positions[i], random_positions[i + 1]]] for i in
                                       range(0, 2 * (positions // 2), 2)]
                    continue

                random_positions = random.sample([*range(positions)],2*(positions//2))
                pos_test_indices.extend([[test_index, [random_positions[i], random_positions[i + 1]]] for i in
                                        range(0, 2 * (positions // 2), 2)])

            self.test_indices = pos_test_indices

            random.shuffle(self.test_indices)
            self.testSize = len(self.test_indices)





            for i, val_index in enumerate(self.val_indices):
                if i==0:
                    random_positions = random.sample([*range(positions)],2*(positions//2))
                    pos_val_indices = [[val_index, [random_positions[i], random_positions[i+1]]] for i in range(0,2*(positions//2),2)]
                    continue

                random_positions = random.sample([*range(positions)],2*(positions//2))
                pos_val_indices.extend([[val_index, [random_positions[i], random_positions[i+1]]] for i in range(0, 2*(positions//2), 2)])

            self.val_indices = pos_val_indices

            random.shuffle(self.val_indices)
            self.valSize = len(self.val_indices)






            for i, train_index in enumerate(self.train_indices):
                if i == 0:

                    random_positions = random.sample([*range(positions)],2*(positions//2))
                    pos_train_indices = [[train_index, [random_positions[i],random_positions[i+1]]] for i in range(0,2*(positions//2),2)]
                    continue

                random_positions = random.sample([*range(positions)],2*(positions//2))
                pos_train_indices.extend([[train_index, [random_positions[i],random_positions[i+1]]] for i in range(0,2*(positions//2),2)])

            self.train_indices = pos_train_indices

            random.shuffle(self.train_indices)
            self.trainSize = len(self.train_indices)



        return simCLR_bags


    def get_loc_nulls(self, bags):
        position = self.shl_args.train_args['gpsPosition']
        null_loc = []
        for i, loc_bag in \
            enumerate(bags['loc_bags'][position]):
            if not loc_bag:
                null_loc.append(i)

        return null_loc

    def sensor_position(self, accTransfer = False, locTransfer = False, randomTree = False, DHMM = False):

        if not locTransfer:
            positions = len(self.shl_args.train_args['positions'])

            for i, test_index in enumerate(self.test_indices):
                if i == 0:
                    pos_test_indices = [[test_index, pos] for pos in range(positions)]
                    continue

                pos_test_indices.extend([[test_index,pos] for pos in range(positions)])


            self.test_indices = pos_test_indices

            if not randomTree:
                random.shuffle(self.test_indices)

            self.testSize = len(self.test_indices)


            if not DHMM:
                for i, val_index in enumerate(self.val_indices):
                    if i == 0:
                        pos_val_indices = [[val_index, pos] for pos in range(positions)]
                        continue

                    pos_val_indices.extend([[val_index, pos] for pos in range(positions)])


                self.val_indices = pos_val_indices

                if not randomTree:
                    random.shuffle(self.val_indices)

                self.valSize = len(self.val_indices)


        if accTransfer or randomTree:
            for i, train_index in enumerate(self.train_indices):
                if i==0:
                    pos_train_indices = [[train_index, pos] for pos in range(positions)]
                    continue

                pos_train_indices.extend([[train_index, pos] for pos in range(positions)])


            self.train_indices = pos_train_indices

            if not randomTree:
                random.shuffle(self.train_indices)
            self.trainSize = len(self.train_indices)

        elif not locTransfer:
            for i, train_index in enumerate(self.train_indices):
                pos = random.randrange(positions)
                if i==0:
                    pos_train_indices = [[train_index, pos]]
                    continue

                pos_train_indices.append([train_index, pos])


            self.train_indices = pos_train_indices


            if not randomTree:
                random.shuffle(self.train_indices)

            self.trainSize = len(self.train_indices)

    # def post_processing_data(self, Model = None):







    def __call__(self, baseline = False, simCLR = False, accTransfer = False, locTransfer = False, randomTree = False, postprocess = False, round = True, DHMM = False):

        if randomTree:

            bags = self.to_bags()

            self.acc_bags , self.lbs_bags , self.loc_bags = bags['acc_bags'] , bags['lbs_bags'] , bags['loc_bags']

            del bags

            self.run_indices = []
            self.split_train_val_test(DHMM=DHMM)

            self.sensor_position(randomTree=True, DHMM=DHMM)

            if DHMM:
                trainX, trainY, trans_mx = self.to_pandas(DHMM=DHMM)

            else:
                trainX, trainY = self.to_pandas(DHMM=DHMM)

            testX, testY = self.to_pandas(is_test=True, DHMM=DHMM)

            if not DHMM:
                valX, valY = self.to_pandas(is_val=True)

                return trainX, trainY, valX, valY, testX, testY

            else:
                return trainX, trainY, testX, testY, trans_mx


        if not randomTree:
            self.init_transformers(
                baseline = baseline,
                simCLR = simCLR,
                accTransfer = accTransfer,
                locTransfer = locTransfer
            )

            bags = self.to_bags()

            self.acc_bags , self.lbs_bags , self.loc_bags = bags['acc_bags'] , bags['lbs_bags'] , bags['loc_bags']

            if locTransfer:
                nulls = self.get_loc_nulls(bags)

            del bags


            self.run_indices = []
            if self.shl_args.train_args['drop_run']:
                self.drop_run()


            self.split_train_val_test(user_seperated=postprocess, firstRound=round)


            if locTransfer:

                self.test_indices = [test_index for test_index in self.test_indices if test_index not in nulls]
                self.val_indices = [val_index for val_index in self.val_indices if val_index not in nulls]
                self.train_indices = [train_index for train_index in self.train_indices if train_index not in nulls]
                self.testSize = len(self.test_indices)
                self.valSize = len(self.val_indices)
                self.trainSize = len(self.train_indices)



            if simCLR:
                criterion = self.shl_args.train_args['simCLR_criterion']
                self.acc_bags = self.get_simCLR_bags(criterion)

            else:
                criterion = None
                self.sensor_position(accTransfer = accTransfer,
                                     locTransfer = locTransfer)



            if self.dynamicWindow and self.padding_method == 'variableLength':

                self.batches = self.to_batches()


            train = self.to_generator(
                is_val=False, simCLR = simCLR,
                accTransfer = accTransfer, locTransfer = locTransfer,
                criterion = criterion
            )
            val = self.to_generator(
                is_val=True, simCLR = simCLR,
                accTransfer = accTransfer, locTransfer = locTransfer,
                criterion = criterion
            )
            test = self.to_generator(
                is_test=True, simCLR = simCLR,
                accTransfer = accTransfer, locTransfer = locTransfer,
                criterion = criterion
            )


            return self.batch_and_prefetch(train,val,test)


