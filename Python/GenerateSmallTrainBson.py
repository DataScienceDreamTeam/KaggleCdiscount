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

    categories = pd.read_csv(Config.SOURCE_CATEGORY_NAMES_CSV)
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
    df.to_csv(Config.DEST_CATEGORY_NAMES_CSV, header=True, index=False)



def generate_small_train_bson_from_random_categories():

    bson_file_iter = bson.decode_file_iter(open(Config.BSON_TRAIN_FILE, "rb"))

    categories = pd.read_csv(Config.SOURCE_CATEGORY_NAMES_CSV)
    categories_index = categories["category_id"].values
    sorted_categories_index = np.sort(categories_index)


    nb_products = 0
    for c, d in enumerate(bson_file_iter):
        
        category_id = d["category_id"]

        if sorted_categories_index[np.searchsorted(sorted_categories_index, category_id)] == category_id:
            nb_products = nb_products +1

        if nb_products > 0 and nb_products % 10000 == 0:
            print(nb_products)



generate_random_categories()
generate_small_train_bson_from_random_categories()