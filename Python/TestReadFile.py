import numpy as np
import pandas as pd
import bson 
import random
import time


BSON_TRAIN_FILE = "/datadrive/KaggleCdiscount/train.bson"


t1 = time.time()
print("start TestReadFile")
print("BSON_TRAIN_FILE : %s" % BSON_TRAIN_FILE)

bson_file_iter = bson.decode_file_iter(open(BSON_TRAIN_FILE, "rb"))

for c, d in enumerate(bson_file_iter):
    product_id = d["_id"]

    for index_image, image in enumerate(d["imgs"]):
        pass

t2 = time.time()
print("end TestReadFile")
total_time = t2 - t1
print(total_time)