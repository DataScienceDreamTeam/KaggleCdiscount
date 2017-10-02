import numpy as np
import pandas as pd
import bson 
import random
import time
import io
import Config
from skimage.data import imread
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

#Test simple logistic regression with scikit learn (No GPU)
#Cannot do it with all data on laptop because all the dataset has to be in RAM
#Trying on laptop to see if the code works and then run it with the VM


def readFileGetXY(fileName):

    print("start readFileGetXY %s" % fileName)
    bson_file_iter = bson.decode_file_iter(open(fileName, "rb"))

    X = []
    Y = []

    for c, d in enumerate(bson_file_iter):
        product_id = d["_id"]
        category_id = d["category_id"]

        print(product_id)
        for index_image, image in enumerate(d["imgs"]):

            picture = imread(io.BytesIO(image['picture']))
            X.append(picture.reshape(-1))
            Y.append(category_id)

    return X,Y

def save_model_parameters():


    pass

def runLogisticRegression():

    t1 = time.time()
    print("start TestLogisticRegressionScikit")

    train_X_results,train_Y_categories = readFileGetXY(Config.BSON_SMALL_TRAIN_FILE)
    validation_X_results,validation_Y_categories = readFileGetXY(Config.BSON_SMALL_VALIDATION_FILE)


    print(len(train_X_results))
    print(len(train_Y_categories))

    print(len(validation_X_results))
    print(len(validation_Y_categories))
    
   
    lb = preprocessing.LabelBinarizer()
    print(lb.fit_transform(train_Y_categories))
    print(lb.classes_)
    print(lb.fit_transform(validation_Y_categories))
    print(lb.classes_)
    

    model = LogisticRegression()
    print("start fit")
    model.fit(train_X_results,train_Y_categories)
    print("end fit")

    prediction = model.predict(validation_X_results)
    nb_prediction_ok = 0
    
    for index, value in enumerate(prediction):
        if validation_Y_categories[index] == value:
            nb_prediction_ok = nb_prediction_ok + 1

    accuracy = nb_prediction_ok / len(prediction)
    print("accuracy = %f" % accuracy)

    model_parameters = model.get_params()
    
    t2 = time.time()
    print("end TestLogisticRegressionScikit")
    total_time = t2 - t1
    print(total_time)



runLogisticRegression()


