import numpy as np
import pandas as pd
import bson 
import random
import time
import io
import sys

from skimage.data import imread

import Config

def ReadAllFile():
    t1 = time.time()
    print("Start ReadAllFile")
    print("BSON_TRAIN_FILE : %s" % Config.BSON_TRAIN_FILE)

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

    nb_products_read = 0
    for c, d in enumerate(bson_file_iter):
        product_id = d["_id"]

        for index_image, image in enumerate(d["imgs"]):
            pass

        nb_products_read = nb_products_read + 1
        if not Config.MAX_PRODUCTS_TO_PROCESS is None: 
            if nb_products_read == Config.MAX_PRODUCTS_TO_PROCESS:
                break

    t2 = time.time()
    print("End ReadAllFile")
    total_time = t2 - t1
    print(total_time)

def ReadAllFileOpenImages():
    t1 = time.time()
    print("Start ReadAllFileOpenImages")
    print("BSON_TRAIN_FILE : %s" % Config.BSON_TRAIN_FILE)

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

    nb_products_read = 0
    for c, d in enumerate(bson_file_iter):
        product_id = d["_id"]

        for index_image, image in enumerate(d["imgs"]):
            picture = imread(io.BytesIO(image['picture']))

        nb_products_read = nb_products_read + 1
        if not Config.MAX_PRODUCTS_TO_PROCESS is None: 
            if nb_products_read == Config.MAX_PRODUCTS_TO_PROCESS:
                break

    t2 = time.time()
    print("end ReadAllFileOpenImages")
    total_time = t2 - t1
    print(total_time)

def ReadAllFileOpenImagesStoreInArray():

    t1 = time.time()
    print("Start ReadAllFileOpenImagesStoreInArray")
    print("BSON_TRAIN_FILE : %s" % Config.BSON_TRAIN_FILE)

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

    results = []
    nb_products_read = 0
    for c, d in enumerate(bson_file_iter):
        product_id = d["_id"]

        for index_image, image in enumerate(d["imgs"]):
            picture = imread(io.BytesIO(image['picture']))
            results.append(picture)

        nb_products_read = nb_products_read + 1
        if not Config.MAX_PRODUCTS_TO_PROCESS is None: 
            if nb_products_read == Config.MAX_PRODUCTS_TO_PROCESS:
                break

    t2 = time.time()
    print("end ReadAllFileOpenImagesStoreInArray")
    total_time = t2 - t1
    print(total_time)

print(sys.argv)
print("Environment = %s" % Config.ENVIRONMENT)
print("Max products to process = %s" % Config.MAX_PRODUCTS_TO_PROCESS)
ReadAllFile()
ReadAllFileOpenImages()
ReadAllFileOpenImagesStoreInArray()