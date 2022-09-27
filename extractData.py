import shutil

import numpy as np
import os
from configParser import Parser
from dataParser import dataParser

class extractData:
    def __init__(self ,args = None):

        if not args:
            parser = Parser()
            self.shl_args = parser.get_args()

        else:
            self.shl_args = args

        self.path_data = os.path.join(
            self.shl_args.data_args['path'],
            'filteredData'
        )

        path_config = os.path.join(
            self.shl_args.data_args['path'],
            'data_config.yaml'
        )

        if os.path.exists(self.path_data) and os.path.exists(path_config):
            dp = dataParser()
            self.args = dp(path_config)
            print('Found Data')
            self.found = True

        else:
            print('No filteredData folder or config file')
            self.found = False



    def __call__(self,
                 delete_dst = False,
                 delete_tmp = False,
                 delete_final = False):


        if not self.found:
            return


        if delete_dst:

            z = os.path.join(self.shl_args.data_args['path'] , 'dstData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

        if delete_tmp:

            z = os.path.join(self.shl_args.data_args['path'] , 'tmpFolder')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

        if delete_final:

            z = os.path.join(self.shl_args.data_args['path'] , 'finalData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))


        location = []
        for position in ['Torso' ,'Hips' ,'Bag' ,'Hand']:


            loc_mmap_path = os.path.join(
                self.path_data,
                'location_' + position +'.mmap'
            )

            pos_loc = np.memmap(
                filename=loc_mmap_path,
                mode='r+',
                shape=self.get_shape(self.args.shapes['location'][position]),
                dtype=np.float64
            )

            location.append(pos_loc)

        acc_mmap_path = os.path.join(
            self.path_data,
            'acceleration.mmap'
        )

        acceleration = np.memmap(
            filename=acc_mmap_path,
            mode='r+',
            shape=self.get_shape(self.args.shapes['acceleration']),
            dtype=np.float64
        )

        lbs_mmap_path = os.path.join(
            self.path_data,
            'labels.mmap'
        )

        labels = np.memmap(
            filename=lbs_mmap_path,
            mode='r+',
            shape=self.get_shape(self.args.shapes['labels'], is_lbs=True),
            dtype=np.int64
        )

        return acceleration , labels , location


    def take_user_day(self, x, u, d):
        ret = []
        stop = False
        for el in x:

            if el[0, -3] == u and el[0, -2] == d:
                stop = True
                ret.append(el)

            elif stop:
                break

        return np.array(ret)

    def get_shape(self ,x ,is_lbs=False):
        if not is_lbs:
            return (x['samples'] ,x['duration'] ,x['channels'])

        else:
            return (x['samples'] ,x['channels'])