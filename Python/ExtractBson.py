import random
import os
import io
import itertools
import time

import numpy as np
import pandas as pd
import bson 

from skimage.data import imread
from skimage.io import imsave
import multiprocessing as mp
import Config

class ExtractBson:
    """Read the cdiscount bson files (train & test), generate the csv and extract all
    pictures on file system"""

    def __init__(self):

        np.random.seed(Config.NUMPY_RANDOM_SEED)

    def get_image_file_name(self, product_id, index_image):
        return f'{product_id:08}'"_"f'{index_image:02}'".png"
    
    def extract_bson(self):

        nb_pictures_extracted = 0
        train = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

        for c,d in enumerate(train):

            product_id = d["_id"]
            category_id = d["category_id"]

            stop = False
            for index_image, image in enumerate(d["imgs"]):
                with open(Config.WRITE_FILE_DIRECTORY + "\\" + str(c) +"_" + str(index_image) + ".jpg","wb") as new_jpg:
                    new_jpg.write(image["picture"])

                nb_pictures_extracted = nb_pictures_extracted + 1

                if nb_pictures_extracted % 100000 == 0:
                    print(nb_pictures_extracted)

                if Config.MAX_PICTURES_TO_EXTRACT is not None and nb_pictures_extracted == Config.MAX_PICTURES_TO_EXTRACT:
                    stop = True
                    break

            if stop == True:
                break


extractor = ExtractBson()
extractor.extract_bson()
