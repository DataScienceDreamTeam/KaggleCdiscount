import random
import os
import io
import itertools

import numpy as np
import pandas as pd
import bson 

from skimage.data import imread
from skimage.io import imsave
import multiprocessing as mp

class ExtractBson:
    """Read the cdiscount bson files (train & test), generate the csv and extract all
    pictures on file system"""

    def __init__(self):

        self.bson_filepath = "G:\KaggleCdiscount\\train.bson"
        self.extract_directory = "G:\\KaggleCdiscountTrainSplit"
        self.extract_directory_multiprocess = "G:\\KaggleCdiscountTrainSplitMultiProcess"      

    def get_image_file_name(self, product_id, index_image):
        return f'{product_id:08}'"_"f'{index_image:02}'".png"
    
    def run(self):

        print("start run")
        bson_file_iter = bson.decode_file_iter(open(self.bson_filepath, "rb"))

        for c, d in enumerate(bson_file_iter):

            product_id = d["_id"]
            
            for index_image, image in enumerate(d["imgs"]):

                picture = imread(io.BytesIO(image['picture']))
                file_name = self.get_image_file_name(product_id, index_image)
                imsave(self.extract_directory + "\\" + file_name, picture)

            if (c % 100000 == 0):
                print(c)

        print("end run")

    def run_multiprocess(self):

        if __name__ == '__main__':
            print("start run_multiprocess")

            bson_file_iter = bson.decode_file_iter(open(self.bson_filepath, "rb"))
            pool = mp.Pool(mp.cpu_count() * 4)

            #7069896 products, 36 * 200000 = 7200000
            for k in range(36):    
                print("processing, k = %i" % k)
                data_slice = itertools.islice(bson_file_iter, 200000)
                result = pool.map_async(self.process,data_slice)

                while not result.ready():
                    print("wait for, k = %i" % k)
                    result.wait(1000)
                                
            pool.close()
            pool.join()

            print("end run_multiprocess")

    def process(self, d):

        product_id = d["_id"]
        for index_image, image in enumerate(d["imgs"]):

            picture = imread(io.BytesIO(image['picture']))
            file_name = self.get_image_file_name(product_id, index_image)
            imsave(self.extract_directory_multiprocess + "\\" + file_name, picture)
            
    @staticmethod
    def test_imread_imsave(file_name, picture):
        """test if all the pixel values are the same between a file and an array to ensure there
         is no data loss / image compression due to imsave in png format"""

        read = imread(file_name)
        print(read.shape)
        print(picture.shape)

        nb_equals = 0
        nb_different = 0
        for i in range(0, read.shape[0], 1):
            for j in range(0, read.shape[1], 1):
                for k in range(0, read.shape[2], 1):
                    if read[i][j][k] == picture[i][j][k]:
                        nb_equals = nb_equals + 1
                    else:
                        nb_different = nb_different + 1


        print(nb_equals)
        print(nb_different)
        print(nb_equals / (nb_equals + nb_different))

extractor = ExtractBson()
extractor.run_multiprocess()