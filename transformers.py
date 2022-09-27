import numpy as np

class CategoricalTransform:
    def __init__(self):
        self.encoding=np.zeros((8,8))
        for i in range(8):
            self.encoding[i,i] = 1.

    def __call__(self, label):
        return self.encoding[label-1]


class TemporalAccTransform:
    def __init__(self,
                 n_bags=None,
                 pos_name_list=None,
                 signal_name_list=None,
                 fusion='Seperate'):

        self.n_bags = n_bags

        if pos_name_list == None:
            self.pnl = ['Torso', 'Hips', 'Bag', 'Hand']

        else:
            self.pnl = pos_name_list

        if signal_name_list == None:
            self.snl = ['Acc_x', 'Acc_y', 'Acc_z', 'Acc_norm']

        else:
            self.snl = signal_name_list

        self.fusion = fusion

        fusion_list = [
            'Depth',
            'Seperate'
        ]

        if not self.fusion in fusion_list:
            print(self.fusion + ' doesnt exist')
            print('choose from the following')
            print(fusion_list)

        self.positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }

        for pos_name in self.pnl:
            if not pos_name in self.positions:
                print(pos_name + ' doesnt exist')
                print('choose one of the following:')
                print(self.positions)

        self.base_acc_signals = {
            'Acc_x': 0,
            'Acc_y': 1,
            'Acc_z': 2
        }

        self.sec_acc_signals = [
            'Acc_norm',
            'Acc_theta',
            'Acc_phi'
        ]

        for signal_name in signal_name_list:
            if not signal_name in self.base_acc_signals and \
                    not signal_name in self.sec_acc_signals:
                print(signal_name + ' doesnt exist')
                print('choose from the following')
                print(self.base_acc_signals)
                print(self.sec_acc_signals)

    def get_shape(self, args):

        self.length = args.data_args['accDuration']
        self.channels = len(args.train_args['acc_signals'])

        if self.fusion == 'Depth':
            if self.n_bags == 1:
                return (self.length, self.channels)

            return (self.n_bags, self.length, self.channels)

    def __call__(self, acceleration):
        signals = {}

        for pos_name in self.pnl:

            if self.fusion == 'Seperate':
                signals[pos_name] = {}

            pos_i = 3 * self.positions[pos_name]

            for signal_index, signal_name in enumerate(self.snl):

                if signal_name in self.base_acc_signals:
                    acc_i = self.base_acc_signals[signal_name]
                    signal_i = pos_i + acc_i
                    signal = acceleration[:, :, signal_i]

                else:
                    if signal_name == 'Acc_norm':

                        signal = np.sqrt(
                            acceleration[:, :, pos_i] ** 2 + \
                            acceleration[:, :, pos_i + 1] ** 2 + \
                            acceleration[:, :, pos_i + 2] ** 2
                        )

                    elif signal_name == 'Acc_theta':
                        xy_norm = np.sqrt(
                            acceleration[:, :, pos_i] ** 2 + \
                            acceleration[:, :, pos_i + 1] ** 2
                        )

                        signal = np.arctan2(xy_norm, acceleration[:, :, pos_i + 2])

                    elif signal_name == 'Acc_phi':

                        signal = np.arctan2(acceleration[:, :, pos_i + 1], acceleration[:, :, pos_i])

                if self.fusion == 'Seperate':
                    signals[pos_name][signal_name] = signal


                elif self.fusion == 'Depth':
                    if signal_index == 0:
                        signals[pos_name] = signal[:, :, np.newaxis]


                    else:
                        signals[pos_name] = np.concatenate((signals[pos_name],
                                                            signal[:, :, np.newaxis]),
                                                           axis=2)
                del signal

            if self.n_bags:
                n_null = self.n_bags - signals[pos_name].shape[0]
                if n_null > 0:
                    if self.fusion == 'Depth':
                        extra_nulls = np.zeros((n_null, self.length, self.channels))
                        signals[pos_name] = np.concatenate((signals[pos_name], extra_nulls),
                                                           axis=0)

                if self.n_bags == 1:
                    signals[pos_name] = signals[pos_name][0,:,:]

        if len(self.pnl)==1:
            return signals[self.pnl[0]]

        return signals


import scipy
from scipy import signal
from scipy import interpolate


class SpectogramAccTransform():

    def __init__(self,
                 n_bags=None,
                 pos_name_list=['Torso', 'Hips', 'Bag', 'Hand'],
                 signal_name_list=['Acc_x', 'Acc_y', 'Acc_z', 'Acc_norm'],
                 fusion='Seperate',
                 freq=20,
                 duration_window=25,
                 duration_overlap=24.5,
                 batch_size=1,
                 log_power=True,
                 out_size=(48, 48)):

        self.temp_tfrm = TemporalAccTransform(pos_name_list=pos_name_list,
                                              signal_name_list=signal_name_list)

        self.duration_window = int(duration_window * freq)
        self.duration_overlap = int(duration_overlap * freq)
        self.batch_size = batch_size
        self.freq = freq
        self.pos_name_list = pos_name_list
        self.signal_name_list = signal_name_list
        self.log_power = log_power
        self.out_size = out_size
        self.fusion = fusion
        self.n_bags = n_bags

        self.fusion_list = [
            'Depth',
            'Time',
            'Frequency',
            'Seperate'
        ]

        if not self.fusion in self.fusion_list:
            print(self.fusion + ' doesnt exist')
            print('choose from the following')
            print(self.fusion_list)

    def log_inter(self,
                  spectrograms,
                  freq, time,
                  out_size):

        samples = spectrograms.shape[0]
        out_f, out_t = out_size

        log_f = np.log(freq + freq[1])  # log between 0.2 Hz and 50.2 Hz

        log_f_normalized = (log_f - log_f[0]) / (log_f[-1] - log_f[0])  # between 0.0 and 1.0
        f = out_f * log_f_normalized

        t_normalized = (time - time[0]) / (time[-1] - time[0])
        t = out_t * t_normalized

        out_spectrograms = np.zeros((samples, out_f, out_t), dtype=np.float64)

        f_i = np.arange(out_f)
        t_i = np.arange(out_t)

        for i, spectrogram in enumerate(spectrograms):
            spectrogram_fn = interpolate.interp2d(t, f, spectrogram, copy=False)
            out_spectrograms[i, :, :] = spectrogram_fn(f_i, t_i)

        f_fn = interpolate.interp1d(f, freq, copy=False)
        t_fn = scipy.interpolate.interp1d(t, time, copy=False)

        f_interpolated = f_fn(f_i)
        t_interpolated = t_fn(t_i)

        return f_interpolated, t_interpolated, out_spectrograms

    def get_shape(self, args):
        self.bags = args.train_args['accBagSize']
        self.channels = len(args.train_args['acc_signals'])
        self.height, self.width = self.out_size
        if self.fusion == 'Depth':

            return (self.bags, self.height, self.width, self.channels)

        elif self.fusion == 'Time':

            self.width *= self.channels
            return (self.bags, self.height, self.width)

        elif self.fusion == 'Frequency':

            self.height *= self.channels
            return (self.bags, self.height, self.width)

    def __call__(self, acceleration):

        signals = self.temp_tfrm(acceleration)

        samples = acceleration.shape[0]
        del acceleration

        last_batch = samples % self.batch_size

        f_out, t_out = self.out_size

        outputs = {}

        for pos_name, signal_names in signals.items():

            if self.fusion == 'Seperate':
                outputs[pos_name] = {}

            for signal_index, signal_name in enumerate(signal_names):

                n_batches = (samples) // self.batch_size + 1
                spectrograms = np.zeros((samples, f_out, t_out), dtype=np.float64)

                for i in range(n_batches):

                    start = i * self.batch_size

                    if i == n_batches - 1:
                        if last_batch > 0:
                            end = start + last_batch
                        else:
                            break



                    else:
                        end = start + self.batch_size

                    batch_signals = signals[pos_name][signal_name][start:end, :]
                    f, t, spectrogram = signal.spectrogram(batch_signals,
                                                           fs=self.freq,
                                                           nperseg=self.duration_window,
                                                           noverlap=self.duration_overlap)

                    _, _, spectrograms[start:end, :, :] = self.log_inter(spectrogram,
                                                                         f,
                                                                         t,
                                                                         self.out_size)

                if self.log_power:
                    np.log(spectrograms + 1e-10, dtype=np.float64, out=spectrograms)

                if self.fusion == 'Seperate':
                    outputs[pos_name][signal_name] = spectrograms

                elif self.fusion == 'Depth':

                    if signal_index == 0:
                        outputs[pos_name] = spectrograms[:, :, :, np.newaxis]

                    else:
                        outputs[pos_name] = np.concatenate((outputs[pos_name],
                                                            spectrograms[:, :, :, np.newaxis]),
                                                           axis=3)

                elif self.fusion == 'Time':
                    if signal_index == 0:
                        outputs[pos_name] = spectrograms

                    else:
                        outputs[pos_name] = np.concatenate((outputs[pos_name],
                                                            spectrograms),
                                                           axis=2)

                elif self.fusion == 'Frequency':
                    if signal_index == 0:
                        outputs[pos_name] = spectrograms

                    else:
                        outputs[pos_name] = np.concatenate((outputs[pos_name],
                                                            spectrograms),
                                                           axis=1)

            if self.n_bags:
                n_null = self.n_bags - outputs[pos_name].shape[0]
                if n_null > 0:
                    if self.fusion == 'Depth':
                        extra_nulls = np.zeros((n_null, self.height, self.width, self.channels))

                    elif self.fusion == 'Time' or self.fusion == 'Frequency':
                        extra_nulls = np.zeros((n_null, self.height, self.width))

                    outputs[pos_name] = np.concatenate((outputs[pos_name], extra_nulls),
                                                       axis=0)



        return outputs


class TemporalLocationTransform:
    def __init__(self,
                 n_bags=None,
                 pos_name_list=None,
                 signals_name_list=None,
                 fusion='Seperate'):

        self.positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }

        self.loc_signals = {
            'Acc': 0,
            'Lat': 1,
            'Long': 2,
            'Alt': 3
        }

        fusion_list = [
            'Seperate',
            'DNN'
        ]

        self.n_bags = n_bags
        self.pos_name_list = pos_name_list

        if self.pos_name_list == None:
            self.pos_name_list = [
                'Torso',
                'Hips',
                'Bag',
                'Hand'
            ]

        self.signals_name_list = signals_name_list

        if self.signals_name_list == None:
            self.signals_name_list = [
                'Acc',
                'Lat',
                'Long',
                'Alt',
                'Distance',
                'Velocity',
                'Acceleration',
                'Walk',
                'Stability'
            ]

        self.fusion = fusion

        if not self.fusion in fusion_list:
            print(self.fusion + ' doesnt exist')
            print('Choose from the following:')
            print(fusion_list)

        for pos_name in pos_name_list:
            if not pos_name in self.positions:
                print(pos_name + ' doesnt exist')
                print('Choose from the following:')
                print(self.positions)

    def get_shape(self, args):

        self.length = args.data_args['locDuration']

        if self.fusion == 'DNN':
            self.totalLength = 0
            for signal_name in args.train_args['loc_signals']:
                if signal_name in self.loc_signals:
                    self.totalLength += self.length

                elif signal_name in ['Distance', 'Velocity']:
                    self.totalLength += self.length - 1

                elif signal_name == 'Acceleration':
                    self.totalLength += self.length - 2

                elif signal_name in ['Walk', 'Stability']:
                    self.totalLength += 1

            if self.n_bags == 1:
                return (self.totalLength)

            return (self.n_bags, self.totalLength)

    def calc_distance(self, p, moment):
        return p[moment] - p[moment - 1]

    def calc_velocity(self, p, t, moment):
        dp = p[moment] - p[moment - 1]
        dt = t[moment] - t[moment - 1]
        return dp / dt

    def calc_acceleration(self, p, t, moment):
        dv = p[moment - 2] - 2 * p[moment - 1] + p[moment]
        dt = (t[moment] - t[moment - 1]) * (t[moment - 1] - t[moment - 2])
        return dv / dt

    def distance(self, pos_location, samples, duration):
        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        z_signal = pos_location[:, :, 3]
        dis_signal = np.zeros((samples, duration - 1))

        for i, (x, y, z, t) in enumerate(zip(
                x_signal,
                y_signal,
                z_signal,
                time_signal
        )):
            for moment in range(1, duration):
                dx = self.calc_distance(x, moment)
                dy = self.calc_distance(y, moment)
                dz = self.calc_distance(z, moment)

                dis_signal[i][moment - 1] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        return dis_signal

    def velocity(self, pos_location, samples, duration):
        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        z_signal = pos_location[:, :, 3]
        vel_signal = np.zeros((samples, duration - 1))

        for i, (x, y, z, t) in enumerate(zip(
                x_signal,
                y_signal,
                z_signal,
                time_signal
        )):
            for moment in range(1, duration):
                Vx = self.calc_velocity(x, t, moment)
                Vy = self.calc_velocity(y, t, moment)
                Vz = self.calc_velocity(z, t, moment)
                V = np.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2)
                vel_signal[i][moment - 1] = V

        return vel_signal

    def acceleration(self, pos_location, samples, duration):
        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        z_signal = pos_location[:, :, 3]
        acc_signal = np.zeros((samples, duration - 2))

        for i, (x, y, z, t) in enumerate(zip(
                x_signal,
                y_signal,
                z_signal,
                time_signal
        )):
            for moment in range(2, duration):
                Ax = self.calc_acceleration(x, t, moment)
                Ay = self.calc_acceleration(y, t, moment)
                Az = self.calc_acceleration(z, t, moment)
                A = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)
                acc_signal[i][moment - 2] = A

        return acc_signal

    def __call__(self, location):

        signals = {}


        for pos_name in self.pos_name_list:

            if self.fusion == 'Seperate':
                signals[pos_name] = {}

            pos_location = location[self.positions[pos_name]]

            samples = pos_location.shape[0]

            for signal_index, signal_name in enumerate(self.signals_name_list):
                if signal_name in self.loc_signals:
                    signal = pos_location[:, :, self.loc_signals[signal_name]]


                else:
                    if signal_name == 'Distance':

                        signal = self.distance(pos_location, samples, self.length)


                    elif signal_name == 'Velocity':

                        signal = self.velocity(pos_location, samples, self.length)

                    elif signal_name == 'Acceleration':

                        signal = self.acceleration(pos_location, samples, self.length)

                    elif signal_name == 'Walk':

                        dis_signal = self.distance(pos_location, samples, self.length)
                        signal = np.zeros((samples, 1))
                        for i, (positions, distances) in enumerate(zip(pos_location, dis_signal)):
                            displacement = np.sum(distances)
                            total_dx = positions[0, 1] - positions[-1, 1]
                            total_dy = positions[0, 2] - positions[-1, 2]
                            total_dz = positions[0, 3] - positions[-1, 3]

                            total_distance = np.sqrt(total_dx ** 2 + total_dy ** 2 + total_dz ** 2)

                            signal[i, 0] = total_distance / displacement

                    elif signal_name == 'Stability':
                        acc_signal = self.acceleration(pos_location, samples, self.length)
                        signal = np.zeros((samples, 1))
                        signal[:, 0] = np.sum(np.log(acc_signal + 1.), axis=1)

                if self.fusion == 'Seperate':
                    signals[pos_name][signal_name] = signal

                elif self.fusion == 'DNN':
                    if signal_index == 0:
                        signals[pos_name] = signal

                    else:
                        signals[pos_name] = np.concatenate((signals[pos_name],
                                                            signal), axis=1)

            if self.n_bags:
                n_null = self.n_bags - samples
                if n_null > 0:
                    if self.fusion == 'DNN':
                        extra_nulls = np.zeros((n_null, self.totalLength))
                        signals[pos_name] = np.concatenate((signals[pos_name], extra_nulls),
                                                           axis=0)

                if self.n_bags == 1:
                    signals[pos_name] = signals[pos_name][0,:]

        if len(self.pos_name_list):
            return signals[pos_name]

        return signals
