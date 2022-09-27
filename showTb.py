

import  tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from tensorboard import program
user = 1
tracking_address = r"C:\Users\chris\PycharmProjects\shlProject\logs_user" + str(user) + '/MIL_tensorboard'  # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()


