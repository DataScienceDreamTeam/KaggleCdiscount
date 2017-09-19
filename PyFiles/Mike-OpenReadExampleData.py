# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

''' *** IMPORTATIONS *** '''

import mlbox as mlb
import bson  # this is installed with the pymongo package
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import io
#%%

data = bson.decode_file_iter(open('D:/My Documents/0-KAGGLE/CDISCOUNT/train_example.bson', 'rb')) #dict_keys(['_id', 'imgs', 'category_id'])
prod_to_category = dict()
images = [] #List where I will append images only

for c, d in enumerate(data): # d are dictionnaries with keys ("_id", "imgs", "category_id")
    product_id = d['_id']
    print("Product id", product_id)

#    print("Id:",product_id)
    category_id = d['category_id'] # This won't be in Test data
    #prod_to_category[product_id] = category_id    
    for e, pic in enumerate(d['imgs']): #e is the index of each picture in imgs (so number of pictures for a given Id is e+1)
        print(e)
        picture = imread(io.BytesIO(pic['picture'])) #The only key in pic is 'picture'
        images.append(picture)
        prod_to_category[c,e] = picture, category_id
        # do something with the picture, etc
prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'pictures'}, inplace=True)
prod_to_category.rename(columns={1: 'category_id'}, inplace=True)

#%%
plt.imshow(prod_to_category["pictures"][0])