
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from SignalsDataset import SignalsDataset
from Baseline import *


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)



SD = SignalsDataset(verbose=True)


fit(SD)

