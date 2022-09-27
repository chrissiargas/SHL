

import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import great_circle
from scipy.signal import firwin,lfilter
from scipy.interpolate import CubicSpline, interp1d, interp2d
from transforms3d.axangles import axangle2mat
from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from scipy.stats import skew, kurtosis

import scipy as sp


def GenerateRandomCurves(N, sigma=0.2, knot=4, xyz=False):
    if not xyz:
        xx = ((np.arange(0,N, (N-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
        x_range = np.arange(N)
        cs_x = CubicSpline(xx[:], yy[:])
        return np.array([cs_x(x_range)]).transpose()

    else:
        xx = (np.ones((3, 1)) * (np.arange(0, N, (N - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, 3))
        x_range = np.arange(N)
        cs_x = CubicSpline(xx[:, 0], yy[:, 0])
        cs_y = CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = CubicSpline(xx[:, 2], yy[:, 2])
        return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()



def DistortTimesteps(N, sigma=0.2, xyz=False):
    if not xyz:
        tt = GenerateRandomCurves(N, sigma)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [(N-1)/tt_cum[-1]]
        tt_cum[:] = tt_cum[:]*t_scale
        return tt_cum

    else:
        tt = GenerateRandomCurves(N, sigma,xyz = xyz)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [(N - 1) / tt_cum[-1, 0], (N - 1) / tt_cum[-1, 1], (N - 1) / tt_cum[-1, 2]]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
        tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
        return tt_cum



def DA_TimeWarp(N, sigma=0.2, xyz=False):
    if not xyz:
        tt_new = DistortTimesteps(N, sigma)
        tt_new = np.squeeze(tt_new)
        x_range = np.arange(N)
        return tt_new, x_range


    else:
        tt_new = DistortTimesteps(N, sigma, xyz)
        x_range = np.arange(N)
        return tt_new, x_range



def DA_Permutation(N, nPerm=4, minSegLength=10, xyz=False):
    if not xyz:

        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, N-minSegLength, nPerm-1))
            segs[-1] = N
            if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                bWhile = False

        return segs

def permutate(signal, N, segs, idx, nPerm = 4):
    pp = 0
    X_new = np.zeros(N)

    for ii in range(nPerm):
        x_temp = signal[segs[idx[ii]]:segs[idx[ii] + 1]]
        X_new[pp:pp + len(x_temp)] = x_temp
        pp += len(x_temp)

    return X_new

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axis_angle_to_rotation_matrix_3d_vectorized(axis,angle))


def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):


    x, y, z = axes

    n = np.sqrt(x * x + y * y + z * z)
    x = x / n
    y = y / n
    z = z / n

    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    return np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])





class CategoricalTransform:
    def __init__(self,drop_run):
        self.drop_run = drop_run

        if self.drop_run:
            self.encoding = np.zeros((7,7),dtype=np.float32)
            for i in range(7):
                self.encoding[i,i] = 1.
        else:
            self.encoding=np.zeros((8,8),dtype=np.float32)
            for i in range(8):
                self.encoding[i,i] = 1.

    def __call__(self, label):
        if self.drop_run and label>3:

            return self.encoding[label-2]

        return self.encoding[label-1]




class TemporalAccTransform:
    def __init__(self, shl_args = None, baseline = False, Seperate = False, simCLR = False):




        self.shl_args = shl_args

        self.n_bags = shl_args.train_args['accBagSize']
        self.highpass = shl_args.train_args['highpass_filter']

        if shl_args.train_args['positions'] == None:
            self.pnl = ['Hand']

        else:
            self.pnl = shl_args.train_args['positions']

        if shl_args.train_args['acc_signals'] == None:
            self.snl = ['Acc_norm']

        else:
            self.snl = shl_args.train_args['acc_signals']

        if not Seperate:
            self.fusion = shl_args.train_args['acc_fusion']

        else:
            self.fusion = 'Seperate'

        self.positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }

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


        self.acc_aug_list = shl_args.train_args['acc_norm_augmentation']
        self.acc_aug_params = shl_args.train_args['acc_norm_aug_params']
        self.acc_xyz_aug_list = shl_args.train_args['acc_xyz_augmentation']
        self.acc_xyz_aug_params = shl_args.train_args['acc_xyz_aug_params']
        self.baseline = baseline
        self.simCLR = simCLR

        self.length = self.shl_args.data_args['accDuration']
        self.channels = len(self.shl_args.train_args['acc_signals'])


    def get_shape(self):


        if self.fusion == 'Depth':
            if self.n_bags == 1 and self.baseline:
                return (self.length, self.channels)

            return (self.n_bags, self.length, self.channels)

    def __call__(self, acceleration, is_train = True, pnl = None, position = None):
        signals = {}

        if not position:
            if self.fusion == 'Seperate':
                bag_pnl = pnl if pnl else self.pnl

            else:
                bag_pnl = random.sample(self.pnl, 1) if not self.simCLR else self.pnl



        else:
            bag_pnl = [position]


        for pos_name in bag_pnl:

            if self.fusion == 'Seperate':
                signals[pos_name] = {}

            pos_i = 3 * self.positions[pos_name]

            if is_train and self.acc_xyz_aug_list:
                acc_xyz = acceleration[:, :, pos_i: pos_i + 3]


                for acc_aug, param in zip(self.acc_xyz_aug_list, self.acc_xyz_aug_params):
                    # f,r = plt.subplots(2,3)
                    # r[0, 0].plot(acc_xyz[0,:,0])
                    # r[0, 1].plot(acc_xyz[0,:,1])
                    # r[0, 2].plot(acc_xyz[0,:,2])
                    if acc_aug == 'Jittering':

                        noise = np.random.normal(0., param, size=acc_xyz.shape[1:])
                        acc_xyz = np.array([acc + noise for acc in acc_xyz])

                    elif acc_aug == 'TimeWarp':

                        tt_new, x_range = DA_TimeWarp(self.length, param)
                        acc_xyz = np.array([np.array(
                            [np.interp(x_range, tt_new, acc[:, orientation]) for orientation in
                             range(3)]).transpose() for acc in acc_xyz])


                    elif acc_aug == 'Permutation':

                        segs = DA_Permutation(self.length, nPerm=param, minSegLength=200)
                        idx = np.random.permutation(param)
                        acc_xyz = np.array(
                            [np.array([permutate(acc[:,orientation], self.length, segs, idx, param) for orientation in range(3)]).transpose()
                             for acc in acc_xyz]
                        )

                    elif acc_aug == 'Rotation':

                        acc_xyz = np.array([
                            DA_Rotation(acc) for acc in acc_xyz])

                    # r[1, 0].plot(acc_xyz[0,:,0])
                    # r[1, 1].plot(acc_xyz[0,:,1])
                    # r[1, 2].plot(acc_xyz[0,:,2])
                    # plt.show()

                acceleration[:, :, pos_i: pos_i + 3] = acc_xyz

            for signal_index, signal_name in enumerate(self.snl):

                if signal_name in self.base_acc_signals:
                    acc_i = self.base_acc_signals[signal_name]

                    signal = acceleration[:, :, acc_i + pos_i]

                else:
                    if signal_name == 'Acc_norm':


                        signal = np.sqrt(np.sum(acceleration[:, :, pos_i:pos_i + 3]**2,
                                                axis = 2))


                        if self.highpass:
                            hp = firwin(513,0.02,pass_zero=False)
                            signal[:] = lfilter(hp, 1.0, signal[:])

                        if is_train and self.acc_aug_list:

                            for acc_aug,param in zip(self.acc_aug_list,self.acc_aug_params):

                                if acc_aug == 'Jittering':

                                    sigma = np.abs(np.random.normal(0., param))
                                    noise = np.random.normal(0., sigma, size=self.length)
                                    signal = np.array([s + noise for s in signal])

                                elif acc_aug == 'TimeWarp':

                                    sigma = np.abs(np.random.normal(0., param))
                                    tt_new, x_range = DA_TimeWarp(self.length,sigma)
                                    signal = np.array([np.interp(x_range, tt_new, s) for s in signal])

                                elif acc_aug == 'Permutation':

                                    segs = DA_Permutation( self.length, nPerm=param, minSegLength=200)
                                    idx = np.random.permutation(param)
                                    signal = np.array([permutate(s, self.length, segs, idx, param) for s in signal])



                    elif signal_name == 'Acc_theta':
                        xy_norm = np.sqrt(np.sum(acceleration[:, :, pos_i:pos_i+2],
                                                 axis=2))

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

            if self.fusion != 'Seperate':
                if self.n_bags:
                    n_null = self.n_bags - signals[pos_name].shape[0]
                    if n_null > 0:
                        if self.fusion == 'Depth':
                            extra_nulls = np.zeros((n_null, self.length, self.channels))
                            signals[pos_name] = np.concatenate((signals[pos_name], extra_nulls),
                                                               axis=0)

                    if self.n_bags == 1 and self.baseline:
                        signals[pos_name] = signals[pos_name][0,:,:]

        if len(bag_pnl)==1 and self.fusion != 'Seperate':
            return signals[bag_pnl[0]]

        return signals


import scipy
from scipy import signal
from scipy import interpolate

class FastFourierTransform():
    def __init__(self,
                 shl_args,
                 baseline = False,
                 simCLR = False):
        self.shl_args = shl_args
        self.pos_name_list = self.shl_args.train_args['positions']
        self.signal_name_list = self.shl_args.train_args['acc_signals']
        self.fusion = self.shl_args.train_args['acc_fusion']
        self.n_bags = self.shl_args.train_args['accBagSize']
        self.baseline = baseline
        self.simCLR = simCLR

        self.temp_tfrm = TemporalAccTransform(shl_args=shl_args,
                                              Seperate=True,
                                              simCLR=simCLR)

        self.length = self.shl_args.data_args['accDuration']
        self.channels = len(self.shl_args.train_args['acc_signals'])

    def get_shape(self):
        if self.fusion == 'Depth':
            if self.n_bags == 1 and self.baseline:
                return (self.length, self.channels)

        return (self.n_bags, self.length, self.channels)


    def __call__(self, acceleration, is_train = True):
        bag_pnl = random.sample(self.pos_name_list, 1) if not self.simCLR else self.pos_name_list

        signals = self.temp_tfrm(acceleration,
                                 is_train = is_train,
                                 pnl = bag_pnl)

        del acceleration

        outputs = {}

        for pos_name, signal_names in signals.items():

            if self.fusion == 'Seperate':
                outputs[pos_name] = {}

            for signal_index, signal_name in enumerate(signal_names):
                complex_fft = np.fft.fft(signals[pos_name][signal_name], axis = 1)
                power_fft = np.abs(complex_fft)
                power_fft[:,0] = 0.
                centered_power_fft = np.fft.fftshift(power_fft, axes = 1)

                if self.fusion == 'Seperate':
                    outputs[pos_name][signal_name] = centered_power_fft

                elif self.fusion == 'Depth':

                    if signal_index == 0:
                        outputs[pos_name] = centered_power_fft[:,:,np.newaxis]

                    else:
                        outputs[pos_name] = np.concatenate(
                            (
                                outputs[pos_name],
                                centered_power_fft[:, :, np.newaxis]
                            ), axis = 2
                        )

            if self.n_bags:
                n_null = self.n_bags - outputs[pos_name].shape[0]
                if n_null > 0:
                    if self.fusion == 'Depth':
                        extra_nulls = np.zeros(
                            (n_null,self.length, self.channels)
                        )

                    outputs[pos_name] = np.concatenate(
                        (outputs[pos_name], extra_nulls),
                        axis=0
                    )

                if self.n_bags == 1 and self.baseline:
                    outputs[pos_name] = outputs[pos_name][0]

            if len(bag_pnl)==1:
                return outputs[bag_pnl[0]]

            return outputs





class SpectogramAccTransform():

    def __init__(self,
                 shl_args,
                 duration_window=5,
                 duration_overlap=4.9,
                 log_power=True,
                 out_size=(48, 48),
                 baseline = False,
                 simCLR = False):



        self.shl_args = shl_args

        self.freq = int(1. / self.shl_args.data_args['smpl_acc_period'])
        self.duration_window = int(duration_window * self.freq)
        self.duration_overlap = int(duration_overlap * self.freq)


        self.pos_name_list = self.shl_args.train_args['positions']
        self.signal_name_list = self.shl_args.train_args['acc_signals']
        self.log_power = log_power
        self.out_size = out_size
        self.fusion = self.shl_args.train_args['acc_fusion']
        self.n_bags = self.shl_args.train_args['accBagSize']
        self.baseline = baseline
        self.simCLR = simCLR
        self.aug_list = self.shl_args.train_args['specto_aug']
        self.aug_list = [] if self.aug_list == None else self.aug_list


        self.temp_tfrm = TemporalAccTransform(shl_args=shl_args,
                                              Seperate=True,
                                              simCLR=simCLR)

        self.length = self.shl_args.data_args['accDuration']

    # def get_grid_locations(self, image_height, image_width):
    #     """Wrapper for np.meshgrid."""
    #
    #     y_range = np.linspace(0, image_height - 1, image_height)
    #     x_range = np.linspace(0, image_width - 1, image_width)
    #     y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
    #     return np.stack((y_grid, x_grid), -1)
    #
    # def expand_to_minibatch(self, np_array, batch_size):
    #     """Tile arbitrarily-sized np_array to include new batch dimension."""
    #     tiles = [batch_size] + [1] * np_array.ndim
    #     return np.tile(np.expand_dims(np_array, 0), tiles)
    #
    # def get_boundary_locations(self, image_height, image_width, num_points_per_edge):
    #     """Compute evenly-spaced indices along edge of image."""
    #     y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
    #     x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
    #     ys, xs = np.meshgrid(y_range, x_range, indexing='ij')
    #     is_boundary = np.logical_or(
    #         np.logical_or(xs == 0, xs == image_width - 1),
    #         np.logical_or(ys == 0, ys == image_height - 1))
    #     return np.stack([ys[is_boundary], xs[is_boundary]], axis=-1)
    #
    # def add_zero_flow_controls_at_boundary(self, control_point_locations,
    #                                         control_point_flows, image_height,
    #                                         image_width, boundary_points_per_edge):
    #
    #     # batch_size = tensor_shape.dimension_value(control_point_locations.shape[0])
    #     batch_size = control_point_locations.shape[0]
    #
    #     boundary_point_locations = self.get_boundary_locations(image_height, image_width,
    #                                                        boundary_points_per_edge)
    #
    #     boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])
    #
    #     type_to_use = control_point_locations.dtype
    #     # boundary_point_locations = constant_op.constant(
    #     #     _expand_to_minibatch(boundary_point_locations, batch_size),
    #     #     dtype=type_to_use)
    #     boundary_point_locations = self.expand_to_minibatch(boundary_point_locations, batch_size)
    #
    #     # boundary_point_flows = constant_op.constant(
    #     #     _expand_to_minibatch(boundary_point_flows, batch_size), dtype=type_to_use)
    #     boundary_point_flows = self.expand_to_minibatch(boundary_point_flows, batch_size)
    #
    #     # merged_control_point_locations = array_ops.concat(
    #     #     [control_point_locations, boundary_point_locations], 1)
    #
    #     merged_control_point_locations = np.concatenate(
    #         [control_point_locations, boundary_point_locations], 1)
    #
    #     # merged_control_point_flows = array_ops.concat(
    #     #     [control_point_flows, boundary_point_flows], 1)
    #
    #     merged_control_point_flows = np.concatenate(
    #         [control_point_flows, boundary_point_flows], 1)
    #
    #     return merged_control_point_locations, merged_control_point_flows
    #
    # def sparse_image_warp_np(self, image,
    #                          source_control_point_locations,
    #                          dest_control_point_locations,
    #                          interpolation_order=2,
    #                          regularization_weight=0.0,
    #                          num_boundary_points=0):
    #
    #     # image = ops.convert_to_tensor(image)
    #     # source_control_point_locations = ops.convert_to_tensor(
    #     #     source_control_point_locations)
    #     # dest_control_point_locations = ops.convert_to_tensor(
    #     #     dest_control_point_locations)
    #
    #     control_point_flows = (
    #             dest_control_point_locations - source_control_point_locations)
    #
    #     clamp_boundaries = num_boundary_points > 0
    #     boundary_points_per_edge = num_boundary_points - 1
    #
    #     # batch_size, image_height, image_width, _ = image.get_shape().as_list()
    #     batch_size, image_height, image_width, _ = list(image.shape)
    #
    #     # This generates the dense locations where the interpolant
    #     # will be evaluated.
    #
    #     grid_locations = self.get_grid_locations(image_height, image_width)
    #
    #     flattened_grid_locations = np.reshape(grid_locations,
    #                                           [image_height * image_width, 2])
    #
    #     # flattened_grid_locations = constant_op.constant(
    #     #     _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)
    #
    #     flattened_grid_locations = self.expand_to_minibatch(flattened_grid_locations, batch_size)
    #
    #     if clamp_boundaries:
    #         (dest_control_point_locations,
    #          control_point_flows) = self.add_zero_flow_controls_at_boundary(
    #             dest_control_point_locations, control_point_flows, image_height,
    #             image_width, boundary_points_per_edge)
    #
    #         # flattened_flows = interpolate_spline.interpolate_spline(
    #         #     dest_control_point_locations, control_point_flows,
    #         #     flattened_grid_locations, interpolation_order, regularization_weight)
    #     flattened_flows = scipy.interpolate.interp1d(
    #         dest_control_point_locations, control_point_flows, kind="cubic", fill_value=0, bounds_error=False)(flattened_grid_locations)
    #
    #     # dense_flows = array_ops.reshape(flattened_flows,
    #     #                                 [batch_size, image_height, image_width, 2])
    #     dense_flows = np.reshape(flattened_flows,
    #                              [batch_size, image_height, image_width, 2])
    #
    #     # warped_image = dense_image_warp.dense_image_warp(image, dense_flows)
    #     warped_image = warp(image, dense_flows)
    #
    #     return warped_image, dense_flows

    def time_warp_init(self,time_warping_param = 5):


        n, v = self.out_size[1], self.out_size[0]

        # Step 1 : Time warping
        # Image warping control point setting.
        # Source
        pt = tf.random.uniform([], time_warping_param, n - time_warping_param, tf.int32)  # radnom point along the time axis
        src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
        src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
        src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
        src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

        # Destination
        w = tf.random.uniform([], -time_warping_param, time_warping_param, tf.int32)  # distance
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
        dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

        # warp
        source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
        dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

        def time_warp(spectrogram):
            spectrogram = np.transpose(spectrogram, (0, 2, 1))[:, :, :, np.newaxis]

            warped_spectrogram, _ = tfa.image.sparse_image_warp(
                tf.convert_to_tensor(spectrogram, dtype=tf.float32),
                source_control_point_locations,
                dest_control_point_locations,
                num_boundary_points=2
            )

            return np.array(np.transpose(warped_spectrogram, (0, 2, 1, 3))[:, :, :, 0])

        return time_warp

    def frequency_masking_init(self, frequency_masking_param = 5, frequency_mask_num = 2):

        n, v = self.out_size[1], self.out_size[0]


        for i in range(frequency_mask_num):
            f = tf.random.uniform([], 0, frequency_masking_param, dtype=tf.int32)
            v = tf.cast(v, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, v-f, dtype=tf.int32)

            mask = tf.concat(
                (
                    tf.ones(shape=(1, 1, n, v - f0 - f)),
                    tf.zeros(shape=(1, 1, n, f)),
                    tf.ones(shape=(1, 1, n, f0))
                ), axis = 3
            )


            if i==0:
                masks = mask

            else:
                masks = tf.concat((masks, mask), axis=0)


        def frequency_masking(spectrogram):


            masked_spectrogram = spectrogram


            for i in range(frequency_mask_num):

                masked_spectrogram = tf.multiply(masked_spectrogram, masks[i])


            return np.array(masked_spectrogram)

        return frequency_masking

    def time_masking_init(self, time_masking_param = 5, time_mask_num = 2):
        n, v = self.out_size[1], self.out_size[0]


        for i in range(time_mask_num):
            t = tf.random.uniform([], 0, time_masking_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, n-t, dtype=tf.int32)

            mask = tf.concat(
                (
                    tf.ones(shape=(1, 1, n-t0-t, v)),
                    tf.zeros(shape=(1, 1, t, v)),
                    tf.ones(shape=(1, 1, t0, v))
                ), axis = 2
            )


            if i==0:
                masks = mask

            else:
                masks = tf.concat((masks, mask), axis=0)


        def time_masking(spectrogram):
            masked_spectrogram = spectrogram
            for i in range(time_mask_num):
                masked_spectrogram = tf.multiply(masked_spectrogram, masks[i])

            return np.array(masked_spectrogram)

        return time_masking

    def augmentation_init(self):
        aug_functions = []

        for aug in self.aug_list:
            if aug == 'timeWarp':
                aug_functions.append(self.time_warp_init())

            if aug == 'frequencyMask':
                aug_functions.append(self.frequency_masking_init())

            if aug == 'timeMask':
                aug_functions.append(self.time_masking_init())

        def augment(spectrogram):
            for aug_function in aug_functions:
                spectrogram = aug_function(spectrogram)

            return spectrogram

        return augment




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

    def get_shape(self):
        self.bags = self.shl_args.train_args['accBagSize']
        self.channels = len(self.shl_args.train_args['acc_signals'])
        self.height, self.width = self.out_size

        if self.fusion == 'Depth':
            if self.n_bags == 1 and self.baseline:
                return (self.height, self.width, self.channels)

            return (self.bags, self.height, self.width, self.channels)

        elif self.fusion == 'Time':

            self.width *= self.channels

            if self.n_bags == 1 and self.baseline:
                return (self.height, self.width, 1)

            return (self.bags, self.height, self.width, 1)

        elif self.fusion == 'Frequency':

            self.height *= self.channels

            if self.n_bags == 1 and self.baseline:
                return (self.height, self.width, 1)

            return (self.bags, self.height, self.width, 1)


    def __call__(self, acceleration, is_train = True, position = None, bagged = False, size = None, stride = None, acc=False):
        if bagged:
            if acc:
                acceleration = np.array([acceleration[0][(size-1)*stride:]])

            else:
                acceleration = np.array([acceleration[0][i*stride:i*stride + self.length] for i in range(size)])

        # original_acc = copy.deepcopy(acceleration)
        if not position:
            bag_pnl = random.sample(self.pos_name_list,1) if not self.simCLR else self.pos_name_list

        else:
            bag_pnl = [position]


        signals = self.temp_tfrm(acceleration,
                                 is_train = is_train,
                                 pnl = bag_pnl)

        # signals_notAug = self.temp_tfrm(
        #     original_acc,
        #     is_train=False,
        #     pnl=bag_pnl
        # )


        # f, r = plt.subplots(2, 4)
        # for pos_name, signal_names in signals.items():
        #     for signal_index, signal_name in enumerate(signal_names):
        #
        #         r[0, signal_index].plot(signals[pos_name][signal_name][0])
        #         r[1, signal_index].plot(signals_notAug[pos_name][signal_name][0])
        # plt.show()

        del acceleration

        outputs = {}

        for pos_name, signal_names in signals.items():
            if is_train:
                augment = self.augmentation_init()

            if self.fusion == 'Seperate':
                outputs[pos_name] = {}

            for signal_index, signal_name in enumerate(signal_names):
                f, t, spectrograms = signal.spectrogram(signals[pos_name][signal_name],
                                                       fs=self.freq,
                                                       nperseg=self.duration_window,
                                                       noverlap=self.duration_overlap)

                # f2, t2, spectrograms2 = signal.spectrogram(signals_notAug[pos_name][signal_name],
                #                                            fs=self.freq,
                #                                            nperseg=self.duration_window,
                #                                            noverlap=self.duration_overlap)

                _, _, spectrograms = self.log_inter(spectrograms,
                                                     f,
                                                     t,
                                                     self.out_size)

                # _, _, spectrograms2 = self.log_inter(spectrograms2,
                #                                      f2,
                #                                      t2,
                #                                      self.out_size)


                if is_train:
                    spectrograms = augment(spectrograms)
                    # spectrograms2 = augment(spectrograms2)

                if self.log_power:
                    np.log(spectrograms + 1e-10, dtype=np.float64, out=spectrograms)
                    # np.log(spectrograms2 + 1e-10, dtype = np.float64, out = spectrograms2)

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
                        outputs[pos_name] = spectrograms[:, : , :, np.newaxis]

                    else:
                        outputs[pos_name] = np.concatenate((outputs[pos_name],
                                                            spectrograms[:, :, :, np.newaxis]),
                                                           axis=1)



            if self.n_bags:
                n_null = self.n_bags - outputs[pos_name].shape[0]
                if n_null > 0:
                    if self.fusion == 'Depth':
                        extra_nulls = np.zeros((n_null, self.height, self.width, self.channels))

                    elif self.fusion == 'Time' or self.fusion == 'Frequency':
                        extra_nulls = np.zeros((n_null, self.height, self.width, 1))

                    outputs[pos_name] = np.concatenate((outputs[pos_name], extra_nulls),
                                                       axis=0)

                if self.n_bags == 1 and self.baseline:
                    outputs[pos_name] = outputs[pos_name][0]

        if len(bag_pnl)==1:
            return outputs[bag_pnl[0]]

        return outputs



class TemporalLocationTransform:
    def __init__(self, shl_args = None, baseline = False):

        self.shl_args = shl_args
        self.earthR = 6372.

        self.haversine = shl_args.train_args['haversine_distance']
        self.location_noise = shl_args.train_args['location_noise']
        self.interp_augment = shl_args.train_args['location_interp_aug']


        self.positions = {
            'Torso': 0,
            'Hips': 1,
            'Bag': 2,
            'Hand': 3
        }

        if shl_args.data_args['gpsSignal']:

            self.loc_signals = {
                'Acc': 0,
                'Lat': 1,
                'Long': 2,
                'Alt': 3,
                'GPS': 4
            }

        else:


            self.loc_signals = {
                'Acc': 0,
                'Lat': 1,
                'Long': 2,
                'Alt': 3
            }


        self.n_bags = shl_args.train_args['locBagSize']
        # self.pos_name_list = self.shl_args.train_args['positions']

        self.pos_name_list = [self.shl_args.train_args['gpsPosition']]

        self.signals_name_list = shl_args.train_args['loc_signals']
        self.features_name_list = shl_args.train_args['loc_features']

        if self.signals_name_list == None:
            self.signals_name_list = [
                'Velocity',
                'Acceleration'
            ]

        if self.features_name_list == None:
            self.features_name_list = []

        self.fusion = shl_args.train_args['loc_fusion']
        self.baseline = baseline


        self.threshold = shl_args.train_args['padding_threshold']
        self.mask = shl_args.train_args['mask']
        self.dynamicWindow = shl_args.data_args['dynamicWindow']
        self.locSampling = shl_args.data_args['locSampling']
        self.second_order = shl_args.train_args['second_order']

        self.padding_method = shl_args.train_args['padding_method']
        self.interpolation = shl_args.train_args['interpolation']


        self.min = True if 'Min' in self.features_name_list else False
        self.max = True if 'Max' in self.features_name_list else False
        self.mean = True if 'Mean' in self.features_name_list else False
        self.var = True if 'Var' in self.features_name_list else False
        self.quantile = True if 'Quantile' in self.features_name_list else False
        self.skewness = True if 'Skewness' in self.features_name_list else False
        self.kurtosis = True if 'Kurtosis' in self.features_name_list else False


    def get_shape(self):

        self.length = self.shl_args.data_args['locDuration']


        if self.fusion in ['LSTM','BidirectionalLSTM','CNNLSTM','FCLSTM']:
            all_signals = self.shl_args.train_args['loc_signals']

            self.channels = len(all_signals)

            if 'Jerk' in all_signals:
                self.totalLength = self.length - 3
            elif 'Acceleration' in all_signals or 'Rotation' in all_signals or 'Walk' in all_signals:
                self.totalLength = self.length - 2
            elif 'Distance' in all_signals or 'Velocity' in all_signals or 'Direction' in all_signals:
                self.totalLength = self.length - 2 if self.second_order else self.length - 1
            elif any(signal in all_signals for signal in self.loc_signals):
                self.totalLength = self.length

            if self.interp_augment:
                self.totalLength -= 1

            if self.n_bags == 1 and self.baseline:

                if self.fusion == 'CNNLSTM':
                    signals_dims = (self.totalLength, self.channels, 1)

                else:

                    if self.padding_method == 'variableLength':

                        signals_dims = (None, self.channels)

                    else:
                        signals_dims =  (self.totalLength, self.channels)

            else:
                if self.padding_method == 'variableLength':

                    signals_dims = (self.n_bags, None, self.channels)

                else:

                    signals_dims = (self.n_bags, self.totalLength, self.channels)

        elif self.fusion == 'DNN':
            self.totalLength = 0
            for signal_name in self.shl_args.train_args['loc_signals']:
                if signal_name in self.loc_signals:
                    self.totalLength += self.length

                elif signal_name in ['Distance', 'Velocity']:
                    self.totalLength += (self.length - 2 if self.second_order else self.length - 1)

                elif signal_name in ['Acceleration', 'Rotation']:
                    self.totalLength += self.length - 2

                elif signal_name in ['Walk', 'Stability']:
                    self.totalLength += 1



            if self.n_bags == 1 and self.baseline:

                signals_dims = (self.totalLength)

            else:
                signals_dims = (self.n_bags, self.totalLength)

        all_features = self.shl_args.train_args['loc_features']
        self.features = 0
        for feature in all_features:
            if feature in ['TotalDistance','TotalVelocity','TotalWalk']:
                self.features += 1

            elif feature in ['Min','Max','Mean','Var','Quantile','Skewness','Kurtosis']:
                self.features += len(self.signals_name_list)

        if self.n_bags == 1 and self.baseline:
            features_dims = (self.features)

        else:
            features_dims = (self.n_bags, self.features)


        return signals_dims, features_dims



    def add_noise(self, acc, lat, lon, alt, moment):
        noise_radius = np.random.normal(0.,
                                        acc[moment] * self.shl_args.train_args['noise_std_factor'])

        noise_theta = np.random.uniform(0., 1.) * np.pi
        noise_phi = np.random.uniform(0., 2.) * np.pi

        noise_lat = noise_radius * np.cos(noise_phi) * np.sin(noise_theta)
        noise_lon = noise_radius * np.sin(noise_phi) * np.sin(noise_theta)
        noise_alt = noise_radius * np.cos(noise_theta)

        m = 180. / ( self.earthR * 1000. * np.pi )
        new_lat = lat[moment] + noise_lat * m
        new_lon = lon[moment] + noise_lon * m / np.cos(lat[moment] * (np.pi / 180.))
        new_alt = alt[moment] + noise_alt
        return np.array([new_lat , new_lon, new_alt])

    def noisy(self, pos_location, duration):

        return np.array([np.array([self.add_noise(acc, x, y, z, moment) for moment in range(duration)]) for acc,x,y,z in zip(
            pos_location[:, :, 0], pos_location[:, :, 1], pos_location[:, :, 2], pos_location[:, :, 3]
        )])


    def calc_haversine_dis(self, lat, lon, alt, moment):

        point1 = (lat[moment-1],lon[moment-1])
        point2 = (lat[moment],lon[moment])
        return math.sqrt(great_circle(point1,point2).m ** 2 + (alt[moment] - alt[moment-1]) ** 2)

    def calc_haversine_vel(self, lat, lon, alt, t, moment):
        hvs_dis = self.calc_haversine_dis(lat,lon,alt,moment)
        return 1000. * hvs_dis / (t[moment] - t[moment-1])

    def calc_haversine_acc(self, lat, lon, alt, t, moment):
        v1 = self.calc_haversine_vel(lat, lon, alt, t, moment-1)
        v2 = self.calc_haversine_vel(lat, lon, alt, t, moment)
        return 1000. * (v2-v1)/(t[moment] - t[moment-1])

    def calc_haversine_jerk(self, lat, lon, alt, t, moment):
        a1 = self.calc_haversine_acc(lat, lon, alt, t, moment)
        a2 = self.calc_haversine_acc(lat, lon, alt, t, moment-1)
        return 1000. * (a2 - a1)/(t[moment] - t[moment-1])



    def calc_bearing(self, lat, lon, t, moment):


        y = np.sin(lon[moment] - lon[moment - 1])
        x = np.cos(lat[moment - 1]) * np.tan(lat[moment]) - \
            np.sin(lat[moment - 1]) * np.cos(lon[moment] - lon[moment-1])

        angle = (np.degrees(np.arctan2(y , x)) + 360) % 360

        return 1000. * angle / (t[moment] - t[moment-1])

    def calc_bearing_rate(self, lat, lon, t, moment):
        angle1 = self.calc_bearing(lat, lon, t, moment-1)
        angle2 = self.calc_bearing(lat, lon, t, moment)
        return 1000. * (angle2-angle1)/(t[moment] - t[moment-1])




    def calc_sinuosity(self, lat, lon, moment):

        point0 = (lat[moment - 2], lon[moment - 2])
        point1 = (lat[moment - 1], lon[moment - 1])
        point2 = (lat[moment], lon[moment])

        d1 = great_circle(point0, point1).m
        d2 = great_circle(point1, point2).m
        d = great_circle(point0, point2).m

        return d / (d1 + d2 + 1e-10)

    def haversine_distance(self, pos_location, samples, duration, second_order = True):
        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        z_signal = pos_location[:, :, 3]

        di = 1 if second_order else 0
        dis_signal = np.zeros((samples, duration - 1 - di))


        for i, (x, y, z, t) in enumerate(zip(
                x_signal,
                y_signal,
                z_signal,
                time_signal
        )):
            for moment in range(1 + di, duration):

                dis_signal[i][moment - 1 - di] = self.calc_haversine_dis(x,y,z,moment)

        return dis_signal


    def haversine_velocity(self, pos_location, samples, duration, second_order = True):
        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        z_signal = pos_location[:, :, 3]

        di = 1 if second_order else 0
        vel_signal = np.zeros((samples, duration - 1 - di))


        for i, (x, y, z, t) in enumerate(zip(
                x_signal,
                y_signal,
                z_signal,
                time_signal
        )):
            for moment in range(1 + di, duration):

                vel_signal[i][moment - 1 - di] = self.calc_haversine_vel(x,y,z,t,moment)

        return vel_signal


    def haversine_acceleration(self, pos_location, samples, duration):
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

                acc_signal[i][moment - 2] = self.calc_haversine_acc(x,y,z,t,moment)

        return acc_signal

    def haversine_jerk(self, pos_location, samples, duration):
        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        z_signal = pos_location[:, :, 3]
        jerk_signal = np.zeros((samples, duration - 3))

        for i, (x, y, z, t) in enumerate(zip(
                x_signal,
                y_signal,
                z_signal,
                time_signal
        )):
            for moment in range(3, duration):
                jerk_signal[i][moment - 3] = self.calc_haversine_jerk(x, y, z, t, moment)

        return jerk_signal


    def bearing(self, pos_location, samples, duration):

        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        bear_signal = np.zeros((samples, duration - 2))

        for i, (x, y, t) in enumerate(zip(
                x_signal,
                y_signal,
                time_signal
        )):
            for moment in range(2, duration):
                bear_signal[i][moment - 2] = self.calc_bearing(x, y, t, moment)

        return bear_signal

    def bearing_rate(self, pos_location, samples, duration):

        time_signal = pos_location[:, :, -1]
        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        BR_signal = np.zeros((samples, duration - 2))

        for i, (x, y, t) in enumerate(zip(
                x_signal,
                y_signal,
                time_signal
        )):
            for moment in range(2, duration):
                BR_signal[i][moment - 2] = self.calc_bearing_rate(x, y, t, moment)

        return BR_signal

    def sinuosity(self, pos_location, samples, duration):

        x_signal = pos_location[:, :, 1]
        y_signal = pos_location[:, :, 2]
        sin_signal = np.zeros((samples, duration - 2))

        for i, (x, y) in enumerate(zip(
                x_signal,
                y_signal
        )):
            for moment in range(2, duration):
                sin_signal[i][moment - 2] = self.calc_sinuosity(x, y, moment)

        return sin_signal

    def cropping(self, location_bag):

        front = True
        end = False

        front_padding = 0
        end_padding = 0

        max_front_pad = 0
        max_end_pad = 0

        for location_window in location_bag:
            for location in location_window:
                if location[0] != -1.:
                    front = False
                    end = True

                elif front:
                    front_padding += 1


                elif end:
                    end_padding += 1

            max_front_pad = max(front_padding, max_front_pad)
            max_end_pad = max(end_padding, max_end_pad)


        cropped_length = self.length - max_front_pad - max_end_pad

        if cropped_length < self.threshold:

            return np.zeros((0,*location_bag.shape[1:])), None, None, self.length


        return np.delete(location_bag, [*[i for i in range(max_front_pad)], *[-i-1 for i in range(max_end_pad)]], 1), \
               max_front_pad, \
               max_end_pad, \
               cropped_length


    def __call__(self, location, is_train = True):

        signals = {}
        features = {}



        for pos_name in self.pos_name_list:

            if self.fusion == 'Seperate':
                signals[pos_name] = {}


            if self.locSampling == 'labelBased':
                pos_location = location[self.positions[pos_name]][:, :, :-2]

            else:
                pos_location = location[self.positions[pos_name]]


            if self.dynamicWindow:
                pos_location, front_pad, end_pad, length = self.cropping(pos_location)

            else:
                length = self.length

            # f,r = plt.subplots(2)
            #
            # r[0].scatter(pos_location[0,:,1],pos_location[0,:,2],c = 'r',alpha=0.5)
            # r[0].plot(pos_location[0, :, 1], pos_location[0, :, 2])



            if np.size(pos_location) and self.interp_augment:
                if is_train:
                    shift = np.random.uniform(0., 1., size=())
                    x_range = np.arange(length)
                    tt_new = x_range + shift
                    tt_new[-1] = x_range[-1]
                    pos_location[:,:,1:4] = np.array([np.array([
                        scipy.interpolate.interp1d(x_range, xyz[:, orientation])(tt_new) for orientation in range(3)
                    ]).transpose() for xyz in pos_location[:, :, 1:4]])

                pos_location = pos_location[:,:-1]
                length -= 1

            # r[0].scatter(pos_location[0, :, 1], pos_location[0, :, 2], c = 'b', alpha=0.5)
            #
            #
            # r[0].axis('scaled')
            #
            # r[1].plot(x_range, x_range, 'r+')
            # r[1].plot(x_range, tt_new, 'bo')
            # plt.show()


            samples = pos_location.shape[0]



            # f,r = plt.subplots(2,2)
            #
            # r[0,0].scatter(pos_location[0,:,1],pos_location[0,:,2],alpha=0.5)
            # r[0, 0].plot(pos_location[0, :, 1], pos_location[0, :, 2])
            # r[0, 0].axis('scaled')
            # vels = self.turn(pos_location, samples, length)
            # r[0,1].plot(np.arange(self.totalLength),vels[0])

            if self.location_noise and is_train and np.size(pos_location):
                pos_location[:,:,1:4] = self.noisy(pos_location, length)

            # r[1, 0].scatter(pos_location[0, :, 1], pos_location[0, :, 2], alpha=0.5)
            #
            # r[1, 0].plot(pos_location[0, :, 1], pos_location[0, :, 2])
            # r[1, 0].axis('scaled')
            # vels = self.turn(pos_location, samples, length)
            # r[1, 1].plot(np.arange(self.totalLength), vels[0])
            # plt.show()

            # f,r = plt.subplots(2,2)
            #
            #
            # vels = self.haversine_velocity(pos_location, samples, length, self.second_order)
            # r[0,0].plot(np.arange(self.totalLength),vels[0])

            # if self.interp_augment and is_train and np.size(pos_location):
            #     sigma = self.shl_args.train_args['interp_std_factor']
            #     tt_new, x_range = DA_TimeWarp(self.totalLength, sigma)
            #
            #
            # r[1,0].plot(np.arange(self.totalLength),vels[0])
            # r[1,1].plot(x_range, tt_new, 'b-')
            # r[1,1].plot(x_range, x_range, 'r-')
            # plt.show()

            signal_features = None

            for signal_index, signal_name in enumerate(self.signals_name_list):
                if signal_name in self.loc_signals:
                    signal = pos_location[:, :, self.loc_signals[signal_name]]

                else:
                    if signal_name == 'Distance':

                            signal = self.haversine_distance(pos_location, samples, length, self.second_order)


                    elif signal_name == 'Velocity':

                            signal = self.haversine_velocity(pos_location, samples, length, self.second_order)


                    elif signal_name == 'Acceleration':

                        signal = self.haversine_acceleration(pos_location, samples, length)


                    elif signal_name == 'Jerk':

                        signal = self.haversine_jerk(pos_location, samples, length)


                    elif signal_name == 'Bearing':

                        signal = self.bearing(pos_location, samples, length)


                    elif signal_name == 'BearingRate':

                        signal = self.bearing_rate(pos_location, samples, length)


                    elif signal_name == 'Sinuosity':

                        signal = self.sinuosity(pos_location, samples, length)

                    #
                    # if self.interp_augment and is_train and np.size(pos_location):
                    #     signal = np.array([scipy.interpolate.CubicSpline(x_range, s)(tt_new) for s in signal])

                    if self.min:

                        signal_features = np.min(signal, axis=1)[:,np.newaxis] if signal_features is None \
                            else np.concatenate([signal_features, np.min(signal, axis=1)[:,np.newaxis]], axis=1)

                    if self.max:

                        signal_features = np.max(signal, axis=1)[:,np.newaxis] if signal_features is None \
                            else np.concatenate([signal_features, np.max(signal, axis=1)[:,np.newaxis]], axis=1)

                    if self.mean:

                        signal_features = np.mean(signal, axis=1)[:,np.newaxis] if signal_features is None \
                            else np.concatenate([signal_features, np.mean(signal, axis=1)[:,np.newaxis]], axis=1)

                    if self.var:

                        signal_features = np.var(signal, axis=1)[:,np.newaxis] if signal_features is None \
                            else np.concatenate([signal_features, np.var(signal, axis=1)[:,np.newaxis]], axis=1)

                    if self.quantile:

                        signal_features = np.quantile(signal, 0.5, axis=1)[:,np.newaxis] if signal_features is None \
                    else np.concatenate([signal_features, np.quantile(signal, 0.5, axis=1)[:,np.newaxis]], axis=1)

                    if self.skewness:

                        signal_features = skew(signal, axis=1)[:, np.newaxis] if signal_features is None \
                            else np.concatenate([signal_features, skew(signal, axis=1)[:, np.newaxis]], axis=1)

                    if self.kurtosis:

                        signal_features = kurtosis(signal, axis=1)[:, np.newaxis] if signal_features is None \
                            else np.concatenate([signal_features, kurtosis(signal, axis=1)[:, np.newaxis]], axis=1)

                if self.padding_method == 'masking':
                    if np.size(signal) and self.dynamicWindow:

                        signal = np.pad(signal,
                                        [(0,0),(front_pad, end_pad)],
                                        mode='constant',
                                        constant_values=self.mask)

                elif self.padding_method == 'interpolation':
                    if np.size(signal) and self.dynamicWindow:

                       signal = np.array(
                            [
                                interp1d(t, sample, kind=self.interpolation)
                                (np.linspace(t[0],t[-1],self.totalLength))
                                for t,sample in zip(pos_location[:, 2: , -1], signal)
                            ]
                       )

                if self.fusion == 'Seperate':
                    signals[pos_name][signal_name] = signal


                elif self.fusion in ['LSTM','BidirectionalLSTM','FCLSTM']:



                        if signal_index == 0:

                            signals[pos_name] = signal[:,:,np.newaxis]

                        else:

                            if self.padding_method == 'variableLength':
                                signals[pos_name] = np.concatenate(
                                    (signals[pos_name][:, :, :],
                                     signal[:, : ,np.newaxis]),
                                    axis=2
                                )

                            elif self.padding_method in ['masking','interpolation']:
                                signals[pos_name] = np.concatenate(
                                    (signals[pos_name][:, -self.totalLength:, :],
                                     signal[:, -self.totalLength: ,np.newaxis]),
                                    axis=2
                                )

                elif self.fusion == 'CNNLSTM':
                    if signal_index == 0:
                        signals[pos_name] = signal[:, :, np.newaxis, np.newaxis]

                    else:

                        if self.padding_method == 'variableLength':
                            signals[pos_name] = np.concatenate(
                                (signals[pos_name][:, :, :],
                                 signal[:, :, np.newaxis]),
                                axis=2
                            )

                        elif self.padding_method in ['masking', 'interpolation']:
                            signals[pos_name] = np.concatenate(
                                (signals[pos_name][:, -self.totalLength:, :, :],
                                 signal[:, -self.totalLength:, np.newaxis, np.newaxis]),
                                axis=2
                            )

                elif self.fusion == 'DNN':
                    if signal_index == 0:
                        signals[pos_name] = signal

                    else:
                        signals[pos_name] = np.concatenate((signals[pos_name],
                                                            signal), axis=1)

            done = False
            for feature_index, feature_name in enumerate(self.features_name_list):
                if feature_name in ['Min','Max','Mean','Var','Quantile', 'Skewness', 'Kurtosis']:
                    if done:
                        continue

                    else:
                        if feature_index == 0:
                            features[pos_name] = signal_features

                        else:
                            features[pos_name] = np.concatenate([features[pos_name], signal_features], axis= 1)

                        done = True

                else:
                    if feature_name == 'TotalDistance':
                        TotalDistance = np.zeros((samples, 1))
                        for i, positions in enumerate(pos_location):
                            point1 = (positions[0, 1], positions[0, 2])
                            point2 = (positions[-1, 1], positions[-1, 2])
                            TotalDistance[i, 0] = math.sqrt(
                                great_circle(point1, point2).m ** 2 + (positions[-1, 3] - positions[0, 3]) ** 2)

                        if feature_index == 0:
                            features[pos_name] = TotalDistance

                        else:
                            features[pos_name] = np.concatenate([features[pos_name], TotalDistance], axis=1)

                    elif feature_name == 'TotalVelocity':
                        TotalVelocity = np.zeros((samples, 1))
                        for i, positions in enumerate(pos_location):
                            point1 = (positions[0, 1], positions[0, 2])
                            point2 = (positions[-1, 1], positions[-1, 2])
                            TotalVelocity[i, 0] = math.sqrt(
                                great_circle(point1, point2).m ** 2 + (positions[-1, 3] - positions[0, 3]) ** 2
                            ) / (positions[-1, -1] - positions[0, -1])

                        if feature_index == 0:
                            features[pos_name] = TotalVelocity

                        else:
                            features[pos_name] = np.concatenate([features[pos_name], TotalVelocity], axis=1)

                    elif feature_name == 'TotalWalk':
                        dis_signal = self.haversine_distance(pos_location, samples, length, False)

                        TotalWalk = np.zeros((samples, 1))
                        for i, (positions, distances) in enumerate(zip(pos_location, dis_signal)):
                            displacement = np.sum(distances)

                            point1 = (positions[0, 1], positions[0, 2])
                            point2 = (positions[-1, 1], positions[-1, 2])
                            total_distance = math.sqrt(great_circle(point1, point2).m ** 2 + (positions[-1, 3] - positions[0, 3]) ** 2)

                            TotalWalk[i, 0] = total_distance / (displacement+1e-10)

                        if feature_index == 0:
                            features[pos_name] = TotalWalk

                        else:
                            features[pos_name] = np.concatenate([features[pos_name], TotalWalk], axis=1)



            if self.n_bags:
                n_null = self.n_bags - samples

                if n_null > 0:
                    if self.fusion == 'DNN':
                        extra_nulls = np.zeros((n_null, self.totalLength)) + self.mask
                        signals[pos_name] = np.concatenate((signals[pos_name], extra_nulls),
                                                           axis=0)

                    elif self.fusion in ['LSTM','BidirectionalLSTM','FCLSTM']:
                        if self.padding_method == 'variableLength':
                            if signals[pos_name].shape[0]:

                                extra_nulls = np.zeros((n_null, signals[pos_name].shape[1], self.channels)) + self.mask

                                signals[pos_name] = np.concatenate((signals[pos_name], extra_nulls),
                                                                   axis=0)
                            else:

                                extra_nulls = np.zeros((n_null, 1, self.channels)) + self.mask

                                signals[pos_name] = extra_nulls


                        elif self.padding_method in ['masking','interpolation']:
                            extra_nulls = np.zeros((n_null, self.totalLength, self.channels)) + self.mask
                            signals[pos_name] = np.concatenate((signals[pos_name], extra_nulls),
                                                               axis=0)

                    extra_nulls = np.zeros((n_null, self.features)) + self.mask
                    features[pos_name] = np.concatenate((features[pos_name], extra_nulls),
                                                        axis=0)

                if self.n_bags == 1 and self.baseline:
                    signals[pos_name] = signals[pos_name][0]
                    features[pos_name] = features[pos_name][0]

        if len(self.pos_name_list) == 1:

            return signals[self.pos_name_list[0]], features[self.pos_name_list[0]]

        return signals, features
