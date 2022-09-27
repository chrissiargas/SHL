import tensorflow_de

def random_tree_fit(SignalsDataset,
            evaluation = True,
            summary = False,
            verbose = 1):

    train , val , test = SignalsDataset()
    user = SignalsDataset.shl_args.train_args['test_user']