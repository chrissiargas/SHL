import numpy as np
import pickle
import json
import os
import math
import copy
import shutil
import scipy.signal as sn
import yaml
from configParser import Parser
from initData import initData

class preprocessData:
    def __init__(self, args=None,
                 verbose=False,
                 delete_dst=False,
                 delete_tmp=False,
                 delete_final=False,
                 delete_filter=False):

        if not args:
            parser = Parser()
            args = parser.get_args()

        if delete_dst:

            z = os.path.join(args.data_args['path'], 'dstData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if delete_tmp:
            z = os.path.join(args.data_args['path'], 'tmpFolder')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if delete_final:
            z = os.path.join(args.data_args['path'], 'finalData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if delete_filter:
            z = os.path.join(args.data_args['path'], 'filteredData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        self.data = initData(args)
        self.location, self.acceleration, self.labels = self.data(verbose)

        self.verbose = verbose
        self.n_acc = self.data.n_acc
        self.n_loc = self.data.n_loc

        if self.verbose:
            self.print_n()

    def print_n(self):
        try:
            print('ACCELERATION SHAPE')
            print(json.dumps(self.n_acc, indent=4))
            print('')
            print('LOCATION SHAPE')
            print(json.dumps(self.n_loc, indent=4))
            print('')

        except:
            print('ACCELERATION SHAPE')
            print(self.n_acc)
            print('')
            print('LOCATION SHAPE')
            print(self.n_loc)
            print('')

    def get_nans_acc(self, x, check_whole=False):

        # all the NaNs are contentrated in the beginning and the end
        # we trim down our arrays(position acceleration) starting from the same point
        # and also ending at the same point
        # thus the trimmed arrays dont contain any NaN value and are synchronized

        # except from the 2nd day of user 3

        nans = []

        if check_whole:
            for position in ['Bag']:
                curr = x[position]

                stop = False
                for i, el in enumerate(curr):
                    if np.isnan(el).any():
                        stop = True
                        nans.append(i)

                    elif stop:
                        break

        else:
            next_start = 0
            for position in self.data.pos:
                curr = x[position]

                for i, el in enumerate(curr[next_start:]):

                    index = i + next_start
                    if np.isnan(el).any():
                        nans.append(index)

                    else:
                        next_start = index
                        break

            next_start = -1
            for position in self.data.pos:
                curr = x[position]

                for i, el in enumerate(curr[next_start::-1]):
                    index = -i + next_start
                    if np.isnan(el).any():
                        nans.append(index)

                    else:
                        next_start = index
                        break

        return np.array(nans, dtype=np.int64)

    def get_nulls(self, y):

        nulls = []

        for i, el in enumerate(y):
            if el[1] == 0:
                nulls.append(i)

        return nulls

    def get_nans(self, x):
        nans = []

        for i, el in enumerate(x):
            if np.isnan(el).any():
                nans.append(i)

        return nans

    def loc_drop(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        dropNan = self.data.args.data_args['dropnan']
        location = {}

        if dropNan:
            for user, days in self.data.files.items():
                location[user] = {}

                for day in days:
                    location[user][day] = {}

                    for position in self.data.pos:
                        current_loc = self.location[user][day][position]
                        nans = self.get_nans(current_loc)

                        n_after_clean = self.n_loc[user][day][position] - len(nans)

                        tmp_dst_filename = 'user' + user + '_' + day + \
                                           '_' + position + '_location' + '.mmap'

                        tmp_dst_loc = os.path.join(
                            path,
                            tmp_dst_filename
                        )

                        if self.verbose:
                            print(tmp_dst_loc)

                        tmp_mmap_loc = np.memmap(
                            tmp_dst_loc,
                            mode=mode,
                            dtype=np.float64,
                            shape=(n_after_clean, 5)
                        )

                        if not exists:
                            tmp_mmap_loc[:] = np.delete(current_loc, nans, axis=0)

                        location[user][day][position] = tmp_mmap_loc
                        self.n_loc[user][day][position] = n_after_clean

        self.location = location

    def acc_drop(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        dropNull = self.data.args.data_args['dropnull']
        dropNan = self.data.args.data_args['dropnan']

        acceleration = {}
        labels = {}
        for user, days in self.data.files.items():

            acceleration[user] = {}
            labels[user] = {}

            for day in days:

                acceleration[user][day] = {}
                labels[user][day] = {}

                nulls = []
                nans = []

                current_acc = self.acceleration[user][day]
                current_lbs = self.labels[user][day]

                if dropNan:
                    nans = self.get_nans_acc(current_acc)

                if dropNull:
                    nulls = self.get_nulls(current_lbs)

                drop = np.concatenate((nans, nulls), axis=0).astype(int)

                n_after_clean = self.n_acc[user][day]['Torso'] - len(nans) - len(nulls)

                for position in self.data.pos:
                    self.n_acc[user][day][position] = n_after_clean

                    tmp_dst_filename = 'user' + user + '_' + day + \
                                       '_' + position + '_motion' + '.mmap'

                    tmp_dst_acc = os.path.join(
                        path,
                        tmp_dst_filename
                    )

                    if self.verbose:
                        print(tmp_dst_acc)

                    tmp_mmap_acc = np.memmap(
                        tmp_dst_acc,
                        mode=mode,
                        dtype=np.float64,
                        shape=(n_after_clean, 4)
                    )

                    if not exists:
                        tmp_mmap_acc[:] = np.delete(current_acc[position],
                                                    drop, axis=0)

                    acceleration[user][day][position] = tmp_mmap_acc

                tmp_dst_filename = 'user' + user + '_' + day + '_labels' + '.mmap'

                tmp_dst_lbs = os.path.join(
                    path,
                    tmp_dst_filename
                )

                if self.verbose:
                    print(tmp_dst_lbs)

                tmp_mmap_lbs = np.memmap(
                    tmp_dst_lbs,
                    mode=mode,
                    dtype=np.float64,
                    shape=(n_after_clean, 2)
                )

                if not exists:
                    tmp_mmap_lbs[:] = np.delete(current_lbs,
                                                drop, axis=0)

                labels[user][day] = tmp_mmap_lbs


        if '3' in self.data.files.keys() and '070717' in self.data.files['3'] and dropNan:
            user = '3'
            day = '070717'

            current_acc = acceleration[user][day]

            acceleration[user]['070717_1'] = {}
            acceleration[user]['070717_2'] = {}

            nans = self.get_nans_acc(current_acc, check_whole=True)

            if (self.verbose):
                print('user' + user + '_' + day + \
                      ' more acceleration nan values:' + str(len(nans)))

            n_after_clean = [nans[0], self.n_acc[user][day]['Torso'] - nans[-1] - 1]

            self.n_acc[user]['070717_1'] = {}
            self.n_acc[user]['070717_2'] = {}

            index = self.data.files['3'].index('070717')
            self.data.files_loc = copy.deepcopy(self.data.files)
            del self.data.files['3'][index]
            self.data.files['3'].insert(index, '070717_2')
            self.data.files['3'].insert(index, '070717_1')

            for position in self.data.pos:

                for segment, sg_day in enumerate(['070717_1', '070717_2']):
                    self.n_acc[user][sg_day][position] = n_after_clean[segment]

                    tmp_dst_filename = 'user' + user + '_' + sg_day + \
                                       '_' + position + '_motion' + '.mmap'

                    tmp_dst_acc = os.path.join(
                        path,
                        tmp_dst_filename
                    )

                    tmp_mmap_acc = np.memmap(
                        tmp_dst_acc,
                        mode=mode,
                        dtype=np.float64,
                        shape=(n_after_clean[segment], 4)
                    )

                    if not exists:
                        if segment == 0:
                            tmp_mmap_acc[:] = current_acc[position][:nans[0]]

                        else:
                            tmp_mmap_acc[:] = current_acc[position][nans[-1] + 1:]

                    acceleration[user][sg_day][position] = tmp_mmap_acc

            current_lbs = labels[user][day]

            labels[user]['070717_1'] = {}
            labels[user]['070717_2'] = {}

            for segment, sg_day in enumerate(['070717_1', '070717_2']):
                tmp_dst_filename = 'user' + user + '_' + sg_day + \
                                   '_labels' + '.mmap'

                tmp_dst_lbs = os.path.join(
                    path,
                    tmp_dst_filename
                )

                tmp_mmap_lbs = np.memmap(
                    tmp_dst_lbs,
                    mode=mode,
                    dtype=np.float64,
                    shape=(n_after_clean[segment], 2)
                )

                if not exists:
                    if segment == 0:
                        tmp_mmap_lbs[:] = current_lbs[:nans[0]]

                    else:
                        tmp_mmap_lbs[:] = current_lbs[nans[-1] + 1:]

                labels[user][sg_day] = tmp_mmap_lbs

        self.acceleration = acceleration
        self.labels = labels

    def get_sampling(self, x, n):

        sampled = []
        threshold = self.data.args.data_args['threshold']
        period = self.data.args.data_args['smpl_loc_period'] * 1000

        i = 0
        j = 1
        minDistance = np.inf
        while True:

            if i == n - 1:
                return np.array(sampled)

            distance = np.abs(period - (x[j, 0] - x[i, 0]))

            if j == n - 1:
                if distance < minDistance:
                    if distance < threshold:

                        if len(sampled) > 0 and sampled[-1] != i:
                            sampled.append(i)

                        sampled.append(j)
                        return np.array(sampled)

                    else:
                        i += 1
                        minDistance = np.inf
                        j = i + 1
                        continue

                else:
                    if minDistance < threshold:
                        if len(sampled) > 0 and sampled[-1] != i:
                            sampled.append(i)

                        sampled.append(j - 1)
                        return np.array(sampled)

                    else:
                        i += 1
                        minDistance = np.inf
                        j = i + 1
                        continue

            if distance < minDistance:
                minDistance = distance
                j += 1

            else:
                if minDistance <= threshold:

                    if len(sampled) == 0:
                        sampled.append(i)

                    elif sampled[-1] != i:
                        sampled.append(i)

                    sampled.append(j - 1)
                    i = j - 1

                else:
                    if sampled[-1] != -1:
                        sampled.append(-1)
                    i += 1

                minDistance = np.inf
                j = i + 1

    def sampling_location(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        location = {}
        for user, days in self.data.files_loc.items():
            location[user] = {}
            for day in days:
                location[user][day] = {}
                for position in self.data.pos:
                    sample_indices = self.get_sampling(
                        x=self.location[user][day][position],
                        n=self.n_loc[user][day][position]
                    )

                    n_after_sampling = sample_indices.shape[0]

                    tmp_dst_filename = 'user' + user + '_' + day + \
                                       '_' + position + '_location' + '.mmap'

                    tmp_dst_loc = os.path.join(
                        path,
                        tmp_dst_filename
                    )

                    if self.verbose:
                        print(tmp_dst_loc)

                    tmp_mmap_loc = np.memmap(
                        tmp_dst_loc,
                        mode=mode,
                        dtype=np.float64,
                        shape=(n_after_sampling, 5)
                    )

                    if not exists:

                        for j, index in enumerate(sample_indices):
                            if index == -1:
                                tmp_mmap_loc[j] = np.zeros(5) - 1.
                                continue

                            tmp_mmap_loc[j] = self.location[user][day][position][index]

                    location[user][day][position] = tmp_mmap_loc
                    self.n_loc[user][day][position] = n_after_sampling

        self.location = location

    def sampling_acceleration_and_labels(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        initial_period = 0.01  # seconds
        sampling_period = self.data.args.data_args['smpl_acc_period']
        step = int(sampling_period // initial_period)

        acceleration = {}
        labels = {}
        for user, days in self.data.files.items():
            acceleration[user] = {}
            labels[user] = {}
            for day in days:
                n_after_sampling = math.ceil(self.n_acc[user][day]['Torso'] / step)

                acceleration[user][day] = {}
                labels[user][day] = {}

                tmp_dst_filename = 'user' + user + '_' + day + '_labels' + '.mmap'

                tmp_dst_lbs = os.path.join(
                    path,
                    tmp_dst_filename
                )

                if self.verbose:
                    print(tmp_dst_lbs)

                tmp_mmap_lbs = np.memmap(
                    tmp_dst_lbs,
                    mode=mode,
                    dtype=np.float64,
                    shape=(n_after_sampling, 2)
                )

                if not exists:

                    for i, j in enumerate(range(0, self.n_acc[user][day]['Torso'], step)):
                        tmp_mmap_lbs[i] = self.labels[user][day][j]

                labels[user][day] = tmp_mmap_lbs

                for position in self.data.pos:

                    tmp_dst_filename = 'user' + user + '_' + day + \
                                       '_' + position + '_motion' + '.mmap'

                    tmp_dst_acc = os.path.join(
                        path,
                        tmp_dst_filename
                    )

                    if self.verbose:
                        print(tmp_dst_acc)

                    tmp_mmap_acc = np.memmap(
                        tmp_dst_acc,
                        mode=mode,
                        dtype=np.float64,
                        shape=(n_after_sampling, 4)
                    )

                    if not exists:
                        if self.data.args.data_args['sampling'] == 'downsampling':
                            for i, j in enumerate(range(0, self.n_acc[user][day][position], step)):

                                tmp_mmap_acc[i] = self.acceleration[user][day][position][j]

                        elif self.data.args.data_args['sampling'] == 'decimation':
                            for i, j in enumerate(range(0, self.n_acc[user][day][position], step)):
                                tmp_mmap_acc[i, 0] = self.acceleration[user][day][position][j, 0]

                            tmp_mmap_acc[:, 1:4] = sn.decimate(
                                x=self.acceleration[user][day][position][:, 1:4],
                                q=step,
                                ftype='fir',
                                axis=0
                            )

                    print(tmp_mmap_acc)
                    acceleration[user][day][position] = tmp_mmap_acc
                    self.n_acc[user][day][position] = n_after_sampling

        self.acceleration = acceleration
        print(self.acceleration)
        self.labels = labels

    def interp_3d(self, pivot_time, location):

        time = location[:, 0]
        acc = location[:, 1]
        lat = location[:, 2]
        long = location[:, 3]
        alt = location[:, 4]

        undefined = 1e6  # to be modified later

        acc = np.interp(pivot_time, time, acc, left=undefined, right=undefined)
        lat = np.interp(pivot_time, time, lat, left=undefined, right=undefined)
        long = np.interp(pivot_time, time, long, left=undefined, right=undefined)
        alt = np.interp(pivot_time, time, alt, left=undefined, right=undefined)

        return acc, lat, long, alt

    def sync_loc(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        pivot = self.data.args.data_args['pivot_position']
        location = {}

        for user, days in self.data.files_loc.items():
            location[user] = {}
            for day in days:
                location[user][day] = {}
                pos_lengths = self.n_loc[user][day]

                if pivot == 'Largest':

                    pivot_pos = max(pos_lengths, key=pos_lengths.get)
                    pivot_length = pos_lengths[pivot_pos]
                    n_after_sync = pivot_length

                elif pivot == 'Smallest':

                    pivot_pos = min(pos_lengths, key=pos_lengths.get)
                    pivot_length = pos_lengths[pivot_pos]
                    n_after_sync = pivot_length

                else:

                    pivot_pos = pivot
                    pivot_length = pos_lengths[pivot_pos]
                    n_after_sync = pivot_length

                pivot_loc = self.location[user][day][pivot_pos]

                for position in self.data.pos:
                    tmp_dst_filename = 'user' + user + '_' + day + \
                                       '_' + position + '_location' + '.mmap'

                    tmp_dst_loc = os.path.join(
                        path,
                        tmp_dst_filename
                    )

                    if self.verbose:
                        print(tmp_dst_loc)

                    tmp_mmap_loc = np.memmap(
                        tmp_dst_loc,
                        mode=mode,
                        dtype=np.float64,
                        shape=(n_after_sync, 5)
                    )

                    if not exists:
                        current_loc = self.location[user][day][position]
                        accs, lats, longs, alts = self.interp_3d(pivot_loc[:, 0], current_loc)

                        for i, (el, acc, lat, long, alt) in \
                                enumerate(zip(pivot_loc, accs, lats, longs, alts)):
                            tmp_mmap_loc[i, 0] = el[0]
                            tmp_mmap_loc[i, 1] = (0. if acc == 1e6 else acc)
                            tmp_mmap_loc[i, 2] = (el[2] if lat == 1e6 else lat)
                            tmp_mmap_loc[i, 3] = (el[3] if long == 1e6 else long)
                            tmp_mmap_loc[i, 4] = (el[4] if alt == 1e6 else alt)

                    location[user][day][position] = tmp_mmap_loc
                    self.n_loc[user][day][position] = n_after_sync

        self.location = location

    def drop(self, path):

        path = os.path.join(
            path,
            'drop'
        )

        if not os.path.exists(path):
            os.makedirs(path)

            self.loc_drop(path, exists=False)
            self.acc_drop(path, exists=False)

            return

        self.loc_drop(path, exists=True)
        self.acc_drop(path, exists=True)

    def sampling(self, path):

        path = os.path.join(
            path,
            'sampling'
        )

        if not os.path.exists(path):
            os.makedirs(path)

            self.sampling_location(path, exists=False)
            self.sampling_acceleration_and_labels(path, exists=False)

            return

        self.sampling_location(path, exists=True)
        self.sampling_acceleration_and_labels(path, exists=True)

    def sync(self, path):

        path = os.path.join(
            path,
            'sync'
        )

        if not os.path.exists(path):
            os.makedirs(path)

            self.sync_loc(path, exists=False)

            return

        self.sync_loc(path, exists=True)

    def modify(self, dropping=True, sampling=True, syncing=False, show_n=False):

        path = self.data.args.data_args['path']

        path = os.path.join(
            path,
            'tmpFolder'
        )

        if not os.path.exists(path):
            os.makedirs(path)

        if dropping:
            if self.verbose or show_n:
                print('DROPPING UNWANTED DATA')
                print('')

            self.drop(path)

            if self.verbose or show_n:
                print('')
                if show_n:
                    self.print_n()

                print('')
                print('------------------------')



        if sampling:
            if self.verbose or show_n:
                print('SAMPLING USING LOWER FREQUENCY')
                print('')

            self.sampling(path)

            if self.verbose or show_n:
                print('')
                if show_n:
                    self.print_n()
                print('')
                print('------------------------')


        #don't use it
        if syncing:
            if self.verbose or show_n:
                print('SYNCHRONIZING LOCATION DATA')
                print('')

            self.sync(path)

            if self.verbose or show_n:
                print('')
                if show_n:
                    self.print_n()
                print('')
                print('------------------------')

    def get_acc_shape(self):
        duration = self.data.args.data_args['accDuration']
        stride = self.data.args.data_args['accStride']

        words = 0
        for user, days in self.data.files.items():
            for day in days:
                channels = 0
                for position in self.data.pos:
                    channels += 3
                    n = self.n_acc[user][day][position]

                extra_words = math.ceil((n - duration + 1) / stride)
                words += max(0, extra_words)

        return (words, duration, channels + 3)  # + 3 for user,day,time

    def get_loc_shape(self):
        duration = self.data.args.data_args['locDuration']
        stride = self.data.args.data_args['locStride']

        words = np.zeros(4, dtype=np.int32)
        for user, days in self.data.files_loc.items():
            for day in days:
                channels = np.zeros(4, dtype=np.int32)
                n = np.zeros(4, dtype=np.int32)
                for i, position in enumerate(self.data.pos):

                    if self.data.args.data_args['useAccuracy']:
                        channels[i] += 4
                    else:
                        channels[i] += 3

                    n_pos = self.n_loc[user][day][position]
                    n_pos = np.ceil((n_pos - duration + 1) / stride)
                    n[i] = max(0, n_pos)

                words += n

        return (words, duration, channels + 3)  # + 3 for user,day,time

    def loc_wordify(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        samples, duration, channels = self.get_loc_shape()

        self.loc_shape = {
            'samples': samples,
            'duration': duration,
            'channels': channels
        }

        final_mmaps = []

        for pos_samples, pos_channel, position in zip(samples, channels, self.data.pos):
            final_filename = 'location' + '_' + position + '.mmap'

            final_loc = os.path.join(
                path,
                final_filename
            )

            if self.verbose:
                print(final_loc)

            final_mmap_loc = np.memmap(
                final_loc,
                mode=mode,
                dtype=np.float64,
                shape=(
                    pos_samples,
                    duration,
                    pos_channel
                )
            )

            if not exists:
                stride = self.data.args.data_args['locStride']
                offset = 0

                for user, days in self.data.files_loc.items():
                    for d, day in enumerate(days):
                        channel = 0

                        current_loc = self.location[user][day][position]
                        n = self.n_loc[user][day][position]
                        words = math.ceil((n - duration + 1) / stride)
                        words = max(0, words)

                        for direction in self.data.loc.keys():

                            if direction == 1 and not self.data.args.data_args['useAccuracy']:
                                continue

                            current_dir = copy.deepcopy(current_loc[:, direction])
                            word_loc = np.lib.stride_tricks.as_strided(
                                current_dir,
                                shape=(words, duration),
                                strides=(stride * 8, 8)
                            )

                            final_mmap_loc[offset:offset + words, :, channel] = word_loc
                            channel += 1

                        time = copy.deepcopy(current_loc[:, 0])
                        word_time = np.lib.stride_tricks.as_strided(
                            time,
                            shape=(words, duration),
                            strides=(stride * 8, 8)
                        )

                        final_mmap_loc[offset:offset + words, :, -3] = int(user)
                        final_mmap_loc[offset:offset + words, :, -2] = d
                        final_mmap_loc[offset:offset + words, :, -1] = word_time
                        offset += words

            final_mmaps.append(final_mmap_loc)

        return final_mmaps

    def acc_lbs_wordify(self, path, exists):

        if exists:
            mode = 'r+'

        else:
            mode = 'w+'

        samples, duration, channels = self.get_acc_shape()

        self.acceleration_shape = {
            'samples': samples,
            'duration': duration,
            'channels': channels
        }

        self.labels_shape = {
            'samples': samples,
            'duration': duration,
            'channels': 4
        }

        final_filename = 'acceleration' + '.mmap'

        final_acc = os.path.join(
            path,
            final_filename
        )

        if self.verbose:
            print(final_acc)

        final_mmap_acc = np.memmap(
            final_acc,
            mode=mode,
            dtype=np.float64,
            shape=(
                samples,
                duration,
                channels
            )
        )

        if not exists:

            stride = self.data.args.data_args['accStride']
            offset = 0
            for user, days in self.data.files.items():

                for d, day in enumerate(days):

                    channel = 0

                    for position in self.data.pos:

                        current_acc = self.acceleration[user][day][position]
                        n = self.n_acc[user][day][position]

                        for direction in self.data.acc.keys():
                            words = math.ceil((n - duration + 1) / stride)
                            words = max(0, words)
                            current_dir = copy.deepcopy(current_acc[:, direction])

                            word_acc = np.lib.stride_tricks.as_strided(
                                current_dir,
                                shape=(words, duration),
                                strides=(stride * 8, 8)
                            )

                            final_mmap_acc[offset:offset + words, :, channel] = word_acc
                            channel += 1

                    time = copy.deepcopy(current_acc[:, 0])
                    word_time = np.lib.stride_tricks.as_strided(
                        time,
                        shape=(words, duration),
                        strides=(stride * 8, 8)
                    )

                    final_mmap_acc[offset:offset + words, :, -3] = int(user)
                    final_mmap_acc[offset:offset + words, :, -2] = d
                    final_mmap_acc[offset:offset + words, :, -1] = word_time
                    offset += words

        final_filename = 'labels' + '.mmap'

        final_lbs = os.path.join(
            path,
            final_filename
        )

        if self.verbose:
            print(final_lbs)

        final_mmap_lbs = np.memmap(
            final_lbs,
            mode=mode,
            dtype=np.int64,
            shape=(
                samples,
                duration,
                4
            )
        )

        if not exists:

            stride = self.data.args.data_args['accStride']
            offset = 0
            for user, days in self.data.files.items():

                for d, day in enumerate(days):

                    current_lbs = self.labels[user][day]

                    for position in self.data.pos:
                        n = self.n_acc[user][day][position]

                    words = math.ceil((n - duration + 1) / stride)
                    words = max(0, words)

                    lbs = copy.deepcopy(current_lbs[:, 1])

                    word_lbs = np.lib.stride_tricks.as_strided(
                        lbs,
                        shape=(words, duration),
                        strides=(stride * 8, 8)
                    )

                    final_mmap_lbs[offset:offset + words, :, 0] = word_lbs

                    time = copy.deepcopy(current_lbs[:, 0])
                    word_time = np.lib.stride_tricks.as_strided(
                        time,
                        shape=(words, duration),
                        strides=(stride * 8, 8)
                    )

                    final_mmap_lbs[offset:offset + words, :, -3] = int(user)
                    final_mmap_lbs[offset:offset + words, :, -2] = d
                    final_mmap_lbs[offset:offset + words, :, -1] = word_time

                    offset += words

        return final_mmap_acc, final_mmap_lbs

    def save_shapes(self):
        shapes = {
            'shapes': {
                'location': {
                    'Torso': {
                        'samples': int(self.loc_shape['samples'][0]),
                        'duration': self.loc_shape['duration'],
                        'channels': int(self.loc_shape['channels'][0])
                    },
                    'Hips': {
                        'samples': int(self.loc_shape['samples'][1]),
                        'duration': self.loc_shape['duration'],
                        'channels': int(self.loc_shape['channels'][1])
                    },
                    'Bag': {
                        'samples': int(self.loc_shape['samples'][2]),
                        'duration': self.loc_shape['duration'],
                        'channels': int(self.loc_shape['channels'][2])
                    },
                    'Hand': {
                        'samples': int(self.loc_shape['samples'][3]),
                        'duration': self.loc_shape['duration'],
                        'channels': int(self.loc_shape['channels'][3])
                    }
                },
                'acceleration': {
                    'samples': self.acceleration_shape['samples'],
                    'duration': self.acceleration_shape['duration'],
                    'channels': self.acceleration_shape['channels']
                },
                'labels': {
                    'samples': self.labels_shape['samples'],
                    'channels': self.labels_shape['channels']
                }
            }
        }

        path = self.data.args.data_args['path']

        config_path = os.path.join(
            path,
            'data_config.yaml'
        )

        with open(config_path, 'w') as yaml_file:
            yaml.dump(shapes, yaml_file, default_flow_style=False)

    def wordify(self, show_n=False):

        path = self.data.args.data_args['path']

        path = os.path.join(
            path,
            'finalData'
        )

        if not os.path.exists(path):
            os.makedirs(path)
            exists = False

        else:
            exists = True

        self.location = self.loc_wordify(path, exists)
        self.acceleration, self.labels = self.acc_lbs_wordify(path, exists)

        if show_n:
            print('PREPARING DATA FOR FEEDING TO THE MODEL')

            print('')
            print('LOCATION SHAPE')
            print("")
            print(self.loc_shape)

            print('')
            print('ACCELERATION SHAPE')
            print("")
            print(self.acceleration_shape)

            print('')
            print('LABELS SHAPE')
            print("")
            print(self.labels_shape)

    def loc_filter(self, path, exists):
        if exists:
            mode = 'r+'
        else:
            mode = 'w+'

        filtered_mmaps_loc = []

        self.loc_shape = {
            'samples': [],
            'duration': 0,
            'channels': []
        }

        for position, pos_name in enumerate(self.data.pos):
            contains_gap = []
            pos_location = self.location[position]
            for i, sample in enumerate(pos_location):
                if np.any(sample == -1):
                    contains_gap.append(i)

            (samples, duration, features) = tuple(pos_location.shape)

            filtered_filename = 'location' + '_' + pos_name + '.mmap'

            filtered_loc = os.path.join(
                path,
                filtered_filename
            )

            if self.verbose:
                print(filtered_loc)

            filtered_mmap_loc = np.memmap(
                filtered_loc,
                mode=mode,
                dtype=np.float64,
                shape=(
                    samples - len(contains_gap),
                    duration,
                    features
                )
            )

            self.loc_shape['samples'].append(samples - len(contains_gap))
            self.loc_shape['duration'] = duration
            self.loc_shape['channels'].append(features)

            if not exists:
                filtered_mmap_loc[:] = np.delete(pos_location, contains_gap, 0)

            filtered_mmaps_loc.append(filtered_mmap_loc)

        return filtered_mmaps_loc

    def get_lb_indices(self):
        output_indices = []
        duration = self.labels.shape[1]
        hard_labelling = self.data.args.data_args['hardLabelling']
        labelling_threshold = self.data.args.data_args['labellingThreshold']
        label_position = self.data.args.data_args['labelPosition']

        if not hard_labelling:
            if not labelling_threshold:
                labelling_threshold = duration

        if label_position:

            if not (label_position >= 0 and label_position < duration):
                print(str(label_position) + " outside labels' range")
                print("labels' range: [0," + str(duration) + "]")
                return

        else:

            label_position = duration // 2

        for i, sample_lbs in enumerate(self.labels):

            label = sample_lbs[label_position, 0]

            if label != 0:
                output_indices.append([i, label_position])

            else:
                if not hard_labelling:

                    step = 1

                    while step <= labelling_threshold:
                        moved_pos = label_position + step
                        if moved_pos < duration and sample_lbs[moved_pos, 0] != 0:
                            output_indices.append([i, moved_pos])
                            break

                        moved_pos = label_position - step
                        if moved_pos >= 0 and sample_lbs[moved_pos, 0] != 0:
                            output_indices.append([i, moved_pos])
                            break

                        step += 1

        return np.array(output_indices)

    def labels_filter(self, path, exists):
        if exists:
            mode = 'r+'
        else:
            mode = 'w+'

        keep_indices = self.get_lb_indices()

        filtered_n = keep_indices.shape[0]
        samples = self.acceleration.shape[0]
        duration = self.acceleration.shape[1]
        features_acc = self.acceleration.shape[2]
        features_lbs = self.labels.shape[2]

        self.acceleration_shape = {
            'samples': samples,
            'duration': duration,
            'channels': features_acc
        }

        self.labels_shape = {
            'samples': filtered_n,
            'channels': features_lbs + 2
        }

        filtered_filename = 'acceleration' + '.mmap'

        filtered_acc = os.path.join(
            path,
            filtered_filename
        )

        if self.verbose:
            print(filtered_acc)

        filtered_mmap_acc = np.memmap(
            filtered_acc,
            mode=mode,
            dtype=np.float64,
            shape=(
                samples,
                duration,
                features_acc
            )
        )

        if not exists:
            for index in range(samples):
                filtered_mmap_acc[index] = self.acceleration[index]


        filtered_filename = 'labels' + '.mmap'

        filtered_lbs = os.path.join(
            path,
            filtered_filename
        )

        if self.verbose:
            print(filtered_lbs)
            print(mode)
            print(exists)
            print(filtered_n)
            print(features_lbs)

        filtered_mmap_lbs = np.memmap(
            filtered_lbs,
            mode=mode,
            dtype=np.int64,
            shape=(
                filtered_n,
                features_lbs + 2
            )
        )

        if not exists:
            for i, index in enumerate(keep_indices):
                label = self.labels[index[0], index[1]]
                filtered_mmap_lbs[i][0] = label[0]
                filtered_mmap_lbs[i][1] = index[0]
                filtered_mmap_lbs[i][2] = index[1]
                filtered_mmap_lbs[i][3:] = label[1:]

        return filtered_mmap_acc, filtered_mmap_lbs

    def filter(self, show_n=False):

        path = self.data.args.data_args['path']

        path = os.path.join(
            path,
            'filteredData'
        )

        if not os.path.exists(path):
            os.makedirs(path)
            exists = False

        else:
            exists = True

        acceleration, labels = self.labels_filter(path, exists)
        location = self.loc_filter(path, exists)

        if show_n:
            print('FILTERING DATA FOR FEEDING TO THE MODEL')

            print('')
            print('LOCATION SHAPE')
            print("")
            print(self.loc_shape)

            print('')
            print('ACCELERATION SHAPE')
            print("")
            print(self.acceleration_shape)

            print('')
            print('LABELS SHAPE')
            print("")
            print(self.labels_shape)

        self.save_shapes()

        return acceleration, labels, location

    def __call__(self, show_n=False):

        self.modify(show_n=show_n)

        self.wordify(show_n=show_n)

        return self.filter(show_n=show_n)




