import numpy as np
import pandas as pd
import bson 
import random
import time
import io
import sys
import multiprocessing as mp
import itertools
from skimage.data import imread
from functools import partial
import Config



def read_train_export():

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))
    bson_filepath =  Config.WRITE_FILE_DIRECTORY + "\\export.bson"
    with open(bson_filepath,"wb") as new_file_bson:
        for c, d in enumerate(bson_file_iter):
            if c == 1:
                break

            for index_image, image in enumerate(d["imgs"]):
                with open(Config.WRITE_FILE_DIRECTORY + "\\" + str(c) +"_" + str(index_image) + ".bmp","wb") as new_jpg:
                    new_jpg.write(image["picture"])

            new_file_bson.write(bson.BSON.encode(d))


def write_bson():

    bson_filepath =  Config.WRITE_FILE_DIRECTORY + "\\myfile.bson"

    myList = [i for i in range(10)]
    myDic = { "key" : myList, "key2" : myList}


    with open(bson_filepath,"wb") as new_file_bson:
        new_file_bson.write(bson.BSON.encode(myDic))
        new_file_bson.write(bson.BSON.encode(myDic))
        

def read_bson():
    bson_filepath =  Config.WRITE_FILE_DIRECTORY + "\\myfile.bson"
    iter = bson.decode_file_iter(open(bson_filepath, 'rb'))

    for c, d in enumerate(iter):

        print(c)
        print(d["key"])
        print(d["key2"])


# write_bson()
# read_bson()
read_train_export()