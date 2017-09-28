import numpy as np
import pandas as pd
import bson 
import random
import time
import io
import Config
from skimage.data import imread

def flatten(tab):
    result = []
    for i in tab:
        for j in i:
            for k in j:
                result.append(k)
    return result

#Test simple logistic regression with scikit learn (No GPU)
#Cannot do it with all data on laptop because all the dataset has to be in RAM
#Trying on laptop to see if the code works and then run it with the VM

t1 = time.time()
print("start TestLogisticRegressionScikit")
print("BSON_TRAIN_FILE : %s" % Config.BSON_TRAIN_FILE)

bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

# input data = all pixels colors : 
# X = 180 x 180 x 3 x nb pictures
# Y = category id for all pictures
#

results = []
categories = []

for c, d in enumerate(bson_file_iter):
    product_id = d["_id"]
    category_id = d["category_id"]

    for index_image, image in enumerate(d["imgs"]):

        picture = imread(io.BytesIO(image['picture']))
        results.append(flatten(picture))
        categories.append(category_id)


# Y = category id for all pictures, need to get the max prediction for each category


t2 = time.time()
print("end TestLogisticRegressionScikit")
total_time = t2 - t1
print(total_time)



