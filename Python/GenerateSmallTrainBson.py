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

def generate_random_categories():

    print("start generate_random_categories")
    categories = pd.read_csv(Config.SOURCE_CATEGORY_NAMES_CSV, encoding = "UTF-8")
    categories_values = categories.values

    np.random.seed(Config.NUMPY_RANDOM_SEED)
    np.random.shuffle(categories_values)
    
    print(Config.NB_CATEGORIES_TO_KEEP)
    index_to_keep = np.arange(Config.NB_CATEGORIES_TO_KEEP)
    select_categories = categories_values[index_to_keep,:]

    for row in select_categories:
        print(row)
    print(len(select_categories))

    df = pd.DataFrame(select_categories, columns = categories.columns)
    df.to_csv(Config.DEST_CATEGORY_NAMES_CSV, header=True, index=False, encoding = "UTF-8")

    print("end generate_random_categories")


def generate_small_train_and_validation_bson_from_categories():

    print("start generate_small_train_and_validation_bson_from_categories")

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

    categories = pd.read_csv(Config.DEST_CATEGORY_NAMES_CSV)
    categories_index = categories["category_id"].values

    nb_products = 0

    items_to_keep = []
    for c, d in enumerate(bson_file_iter):
        
        category_id = d["category_id"]

        if category_id in categories_index:
            nb_products = nb_products +1
            items_to_keep.append(d)
            if nb_products > 0 and nb_products % 50000 == 0:
                print(nb_products)

    nb_items_to_keep =  len(items_to_keep)
    print("nb items to keep : %s" % nb_items_to_keep)
    print("start shuffling items")

    np.random.shuffle(items_to_keep)
    
    index_split_train_validation = int(nb_items_to_keep * Config.PERCENT_TRAIN)

    print("start writing files (train and validation)")

    nb_item_train = 0
    nb_item_validation = 0
    with open(Config.BSON_SMALL_TRAIN_FILE,"wb") as small_train_bson:
        with open(Config.BSON_SMALL_VALIDATION_FILE,"wb") as small_validation_bson:
            
            for index, item in enumerate(items_to_keep):
                if index < index_split_train_validation:
                    small_train_bson.write(bson.BSON.encode(item))
                    nb_item_train = nb_item_train+1
                else:
                    small_validation_bson.write(bson.BSON.encode(item))
                    nb_item_validation = nb_item_validation +1

    print("nb_item_train : %s" % nb_item_train)
    print("nb_item_validation : %s" % nb_item_validation)
    
    print("end generate_smallgenerate_small_train_and_validation_bson_from_categories_train_bson_from_random_categories_not_sorted")


def test_generated_bson():

    train = bson.decode_file_iter(open(Config.BSON_SMALL_TRAIN_FILE, "rb"))

    for c,d in enumerate(train):

        product_id = d["_id"]
        category_id = d["category_id"]

        print(str(product_id) + " " + str(category_id))
        for index_image, image in enumerate(d["imgs"]):
            with open(Config.WRITE_FILE_DIRECTORY + "\\" + str(c) +"_" + str(index_image) + ".jpg","wb") as new_jpg:
                new_jpg.write(image["picture"])



generate_random_categories()
generate_small_train_and_validation_bson_from_categories()

# test_generated_bson()