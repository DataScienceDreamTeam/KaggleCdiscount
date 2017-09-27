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

class ExtractBson:
    """Read the cdiscount bson files (train & test), generate the csv and extract all
    pictures on file system"""

    def __init__(self):

        self.bson_filepath = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
        self.extract_directory = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data"
        self.extract_directory_multiprocess = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\Extract"
        self.csv_train_filepath = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\ExtractCsv\\train.csv"
        self.csv_validation_filepath = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\ExtractCsv\\validation.csv"
        self.random_seed = 0.7860241151479839
        self.np_random_seed= 779493057
        self.percent_to_extract = 0.000001
        self.percent_train = 0.8

        random.seed(self.random_seed)
        np.random.seed(self.np_random_seed)

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

            results = []
            #7069896 products, 36 * 200000 = 7200000
            for k in range(36):    
                print("processing, k = %i" % k)
                data_slice = itertools.islice(bson_file_iter, 200000)
                result = pool.map_async(self.process,data_slice)

                while not result.ready():
                    print("wait for, k = %i" % k)
                    result.wait(1000)

                results.append(result.get())

            self.save_result_to_csv(results)
                     
            pool.close()
            pool.join()

            print("end run_multiprocess")

    def process(self, d):
        results = []

        if random.random() < self.percent_to_extract:
            product_id = d["_id"]
            category_id = d["category_id"]

            for index_image, image in enumerate(d["imgs"]):

                picture = imread(io.BytesIO(image['picture']))
                file_name = self.get_image_file_name(product_id, index_image)
                imsave(self.extract_directory_multiprocess + "\\" + file_name, picture)

                results.append([str(product_id),str(category_id),str(index_image),str(file_name)])

        return results        

    def save_result_to_csv(self, results):

        np.random.shuffle(results)

        with open(self.csv_train_filepath,"w") as new_file_train_csv:
            new_file_train_csv.write("product_id;category_id;img_index;img_filename\n")

            with open(self.csv_validation_filepath,"w") as new_file_validation_csv:
                new_file_validation_csv.write("product_id;category_id;img_index;img_filename\n")

                for tab1 in results:
                    for tab2 in tab1:
                        for line in tab2:
                            if random.random() < self.percent_train:
                                new_file_train_csv.write(";".join(line))
                                new_file_train_csv.write("\n")
                            else:
                                new_file_validation_csv.write(";".join(line))
                                new_file_validation_csv.write("\n")

    #file structure : product_id;category_id;img_index;img_filename
    def generate_csv_file(self, shuffle = False):

        print("start generate_csv_file")
        t1 = time.time()


        nb_products_to_extract = 1000

        dataFrame = pd.DataFrame(columns=('product_id', 'category_id', 'img_index', 'img_filename'))
        bson_file_iter = bson.decode_file_iter(open(self.bson_filepath, "rb"))

        nb_products = 0
        nb_row = 0

        for c, d in enumerate(bson_file_iter):

            product_id = d["_id"]
            category_id = d["category_id"]
            
            for img_index, img in enumerate(d["imgs"]):

                img_filename = self.get_image_file_name(product_id, img_index)

                dataFrame.loc[nb_row] = [product_id, category_id, img_index, img_filename]

                nb_row = nb_row + 1

            nb_products = nb_products + 1
    
            if nb_products == nb_products_to_extract:
                break

        if shuffle == True:
            dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))

        dataFrame.to_csv(self.csv_train_filepath, header=False, index=False)

        t2 = time.time()
        total_time = t2 - t1
        print("total_time : %s" % total_time)

        print("end generate_csv_file")

    def fast_generate_csv_file(self, shuffle = False):

        print("start fast_generate_csv_file")
        t1 = time.time()

        # nb_products_to_extract = 1000

        bson_file_iter = bson.decode_file_iter(open(self.bson_filepath, "rb"))
        
        nb_products = 0
        nb_row = 0

        # Number of products : 7069896 (80% = 5655917)
        # Number of images : 12371293
        nb_products_train = 5655917

        list_of_items_train = []
        list_of_items_validation = []

        for c, d in enumerate(bson_file_iter):

            product_id = d["_id"]
            category_id = d["category_id"]
            
            for img_index, img in enumerate(d["imgs"]):

                img_filename = self.get_image_file_name(product_id, img_index)

                if nb_products <= nb_products_train:
                    list_of_items_train.append([str(product_id), str(category_id), str(img_index), img_filename])
                else:
                    list_of_items_validation.append([str(product_id), str(category_id), str(img_index), img_filename])

                nb_row = nb_row + 1

            nb_products = nb_products + 1
    
            # if nb_products == nb_products_to_extract:
            #     break

        if shuffle == True:
            np.random.shuffle(list_of_items_train)
            np.random.shuffle(list_of_items_validation)
            

        self.save_csv_file(self.csv_train_filepath,list_of_items_train)
        self.save_csv_file(self.csv_validation_filepath,list_of_items_validation)

        t2 = time.time()
        total_time = t2 - t1
        print("total_time : %s" % total_time)

        print("end fast_generate_csv_file")

    def save_csv_file(self, file_path, items):

        with open(file_path,"w") as new_file_csv:
            new_file_csv.write("product_id;category_id;img_index;img_filename\n")      
            for line in items:
                new_file_csv.write(";".join(line))
                new_file_csv.write("\n")

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
# extractor.generate_csv_file(shuffle = True)
#extractor.fast_generate_csv_file(shuffle = True)
extractor.run_multiprocess()