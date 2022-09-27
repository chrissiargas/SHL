import os
import pprint
import time



logdir = os.path.join("results","results_" + time.strftime("%Y%m%d-%H%M%S") + ".txt")



class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(logdir, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


import random
import warnings
import os

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf

import numpy as np
import simCLR

SEED = 0

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


from newSignalDataset import SignalsDataset
from newMILattention import MIL_fit
import sys
import ruamel.yaml



def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:

            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)


def main(logger = True,
         config_test = False,
         regenerate = False,
         all_users = True,
         randomness = True):




    if not randomness:
        set_global_determinism(seed=SEED)


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


    if logger:
        sys.stdout = Logger()




    if config_test:



            for fusion in ['fusion']:
                for test_user in [1, 2, 3]:
                    print(fusion)

                    print(test_user)
                    config_edit('train_args', 'fusion', fusion)

                    config_edit('train_args', 'test_user', test_user)
                    SD = SignalsDataset(regenerate=regenerate)
                    regenerate = False

                    if logger:
                        pprint.pprint(SD.shl_args.data_args)
                        print()
                        pprint.pprint(SD.shl_args.train_args)
                        print()

                    MIL_fit(SD,
                            evaluation=True,
                            summary=False,
                            verbose=1)

            config_edit('train_args', 'fusion', 'MIL')
            for transfer_learning_acc in ['none']:
                for test_user in [1, 2, 3]:
                    print(transfer_learning_acc)

                    print(test_user)
                    config_edit('train_args', 'transfer_learning_acc', transfer_learning_acc)

                    config_edit('train_args', 'test_user', test_user)
                    SD = SignalsDataset(regenerate=regenerate)
                    regenerate = False

                    if logger:
                        pprint.pprint(SD.shl_args.data_args)
                        print()
                        pprint.pprint(SD.shl_args.train_args)
                        print()

                    MIL_fit(SD,
                            evaluation=True,
                            summary=False,
                            verbose=1)

            config_edit('train_args', 'transfer_learning_acc', 'train')

            for spectograms in [False]:
                for test_user in [1, 2, 3]:
                    print(spectograms)

                    print(test_user)
                    config_edit('train_args', 'spectograms', spectograms)

                    config_edit('train_args', 'test_user', test_user)
                    SD = SignalsDataset(regenerate=regenerate)
                    regenerate = False

                    if logger:
                        pprint.pprint(SD.shl_args.data_args)
                        print()
                        pprint.pprint(SD.shl_args.train_args)
                        print()

                    MIL_fit(SD,
                            evaluation=True,
                            summary=False,
                            verbose=1)

            config_edit('train_args', 'spectograms', True)

            for acc_signals,acc_fusion in zip([['Acc_norm']],['Depth']):
                for test_user in [1, 2, 3]:
                    print(acc_signals)
                    print(acc_fusion)

                    print(test_user)
                    config_edit('train_args', 'acc_signals', acc_signals)
                    config_edit('train_args', 'acc_fusion', acc_fusion)

                    config_edit('train_args', 'test_user', test_user)
                    SD = SignalsDataset(regenerate=regenerate)
                    regenerate = False

                    if logger:
                        pprint.pprint(SD.shl_args.data_args)
                        print()
                        pprint.pprint(SD.shl_args.train_args)
                        print()

                    MIL_fit(SD,
                            evaluation=True,
                            summary=False,
                            verbose=1)

            config_edit('train_args', 'acc_signals', ['Acc_x', 'Acc_y', 'Acc_z', 'Acc_norm'])
            config_edit('train_args', 'acc_fusion',  'Frequency')

            for loc_features in [['TotalWalk']]:
                for test_user in [1, 2, 3]:
                    print(loc_features)

                    print(test_user)
                    config_edit('train_args', 'loc_features', loc_features)

                    config_edit('train_args', 'test_user', test_user)
                    SD = SignalsDataset(regenerate=regenerate)
                    regenerate = False

                    if logger:
                        pprint.pprint(SD.shl_args.data_args)
                        print()
                        pprint.pprint(SD.shl_args.train_args)
                        print()

                    MIL_fit(SD,
                            evaluation=True,
                            summary=False,
                            verbose=1)

            config_edit('train_args', 'loc_features',  ['TotalWalk', 'Mean', 'Var'])

            for location_noise in [False]:
                        for test_user in [1, 2, 3]:
                            print(location_noise)

                            print(test_user)
                            config_edit('train_args', 'location_noise', location_noise)

                            config_edit('train_args', 'test_user', test_user)
                            SD = SignalsDataset(regenerate=regenerate)
                            regenerate = False

                            if logger:
                                pprint.pprint(SD.shl_args.data_args)
                                print()
                                pprint.pprint(SD.shl_args.train_args)
                                print()

                            MIL_fit(SD,
                                    evaluation=True,
                                    summary=False,
                                    verbose=1)

            config_edit('train_args', 'location_noise', True)


            for specto_aug in [[]]:
                        for test_user in [1, 2, 3]:
                            print(specto_aug)

                            print(test_user)
                            config_edit('train_args', 'specto_aug', specto_aug)

                            config_edit('train_args', 'test_user', test_user)
                            SD = SignalsDataset(regenerate=regenerate)
                            regenerate = False

                            if logger:
                                pprint.pprint(SD.shl_args.data_args)
                                print()
                                pprint.pprint(SD.shl_args.train_args)
                                print()

                            MIL_fit(SD,
                                    evaluation=True,
                                    summary=False,
                                    verbose=1)

            config_edit('train_args', 'specto_aug', ['frequencyMask', 'timeMask'])

            for test_user in [1,2,3]:
                config_edit('train_args', 'test_user', test_user)
                if test_user == 1 and regenerate:
                    regenerate = True

                else:
                    regenerate = False

                SD = SignalsDataset(regenerate=regenerate)

                if logger:
                    pprint.pprint(SD.shl_args.data_args)
                    print()
                    pprint.pprint(SD.shl_args.train_args)
                    print()

                MIL_fit(SD,
                        evaluation=True,
                        summary=True,
                        verbose=1)


    elif all_users:
        for j in range(10):

            for test_user in [1,2,3]:
                config_edit('train_args', 'test_user', test_user)
                if test_user == 1 and j==0 and regenerate:
                    regenerate = True

                else:
                    regenerate = False

                SD = SignalsDataset(regenerate=regenerate)

                if logger:
                    pprint.pprint(SD.shl_args.data_args)
                    print()
                    pprint.pprint(SD.shl_args.train_args)
                    print()

                MIL_fit(SD,
                        evaluation=True,
                        summary=True,
                        verbose=1)

    else:
        SD = SignalsDataset(regenerate=regenerate)

        if logger:
            pprint.pprint(SD.shl_args.data_args)
            print()
            pprint.pprint(SD.shl_args.train_args)
            print()

        MIL_fit(SD,
                evaluation=True,
                summary=True,
                verbose=1)


if __name__ == "__main__":
    main()





