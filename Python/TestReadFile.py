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

def ReadAllFile(read_mode):
    t1 = time.time()
    print("Start ReadAllFile : %s" % read_mode)
    print("BSON_TRAIN_FILE : %s" % Config.BSON_TRAIN_FILE)

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

    nb_products_read = 0
    results = []
    for c, d in enumerate(bson_file_iter):
        product_id = d["_id"]

        for index_image, image in enumerate(d["imgs"]):

            if read_mode == "READ":
                pass
            elif read_mode == "OPENIMAGE":
                picture = imread(io.BytesIO(image['picture']))
            elif read_mode == "OPENIMAGEANDSAVE":
                picture = imread(io.BytesIO(image['picture']))
                results.append(picture)

        nb_products_read = nb_products_read + 1

        if nb_products_read % 200000 == 0:
            print(nb_products_read)

        if nb_products_read == Config.MAX_PRODUCTS_TO_PROCESS:
            break

    t2 = time.time()
    print("End ReadAllFile")
    total_time = t2 - t1
    print(total_time)

def ReadAllFileMultiProcess(read_mode):

    t1 = time.time()
    print("Start ReadAllFileMultiProcess : %s" % read_mode)
    print("BSON_TRAIN_FILE : %s" % Config.BSON_TRAIN_FILE)

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))
    pool = mp.Pool(mp.cpu_count() * 4)

    nb_products_read = 0

    results_multi = []

    for k in range(Config.MULTI_PROCESS_NB_ITER):   

        data_slice = itertools.islice(bson_file_iter, Config.MULTI_PROCESS_BATCH_SIZE)
        if read_mode == "READ":
            result = pool.map_async(process_read,data_slice)
        elif read_mode == "OPENIMAGE":
            result = pool.map_async(process_openimage,data_slice)
        elif read_mode == "OPENIMAGEANDSAVE":
            result = pool.map_async(process_openimageandsave,data_slice)

        while not result.ready():
            result.wait(1000)
            results_multi.append(result.get())

    pool.close()
    pool.join()

    t2 = time.time()
    print("End ReadAllFileMultiProcess")
    total_time = t2 - t1
    print(total_time)

def process_read(d):
    results = []
    product_id = d["_id"]
    category_id = d["category_id"]
    for index_image, image in enumerate(d["imgs"]):
        pass
    return results  

def process_openimage(d):
    results = []
    product_id = d["_id"]
    category_id = d["category_id"]
    for index_image, image in enumerate(d["imgs"]):
        picture = imread(io.BytesIO(image['picture']))
    return results  

def process_openimageandsave(d):
    results = []
    product_id = d["_id"]
    category_id = d["category_id"]
    for index_image, image in enumerate(d["imgs"]):
        picture = imread(io.BytesIO(image['picture']))
        results.append(picture)
    return results  

if __name__ == '__main__':

    ReadAllFile("READ")
    ReadAllFile("OPENIMAGE")
    ReadAllFile("OPENIMAGEANDSAVE")

    ReadAllFileMultiProcess("READ")
    ReadAllFileMultiProcess("OPENIMAGE")
    ReadAllFileMultiProcess("OPENIMAGEANDSAVE")