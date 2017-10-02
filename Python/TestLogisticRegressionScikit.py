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

def readFileGetXY(fileName):

    print("start readFileGetXY %s" % fileName)
    bson_file_iter = bson.decode_file_iter(open(fileName, "rb"))

    X = []
    Y = []

    for c, d in enumerate(bson_file_iter):
        product_id = d["_id"]
        category_id = d["category_id"]

        for index_image, image in enumerate(d["imgs"]):

            picture = imread(io.BytesIO(image['picture']))
            X.append(picture.reshape(-1))
            Y.append(category_id)

    return X,Y

def save_model_parameters(parameters):


    pass

def get_accuracy(prediction, real_values):

    nb_prediction_ok = 0
    
    for index, value in enumerate(prediction):
        if real_values[index] == value:
            nb_prediction_ok = nb_prediction_ok + 1

    accuracy = nb_prediction_ok / len(prediction)

    return accuracy

def runLogisticRegression():

    t1 = time.time()
    print("start TestLogisticRegressionScikit")

    train_X_results,train_Y_categories = readFileGetXY(Config.BSON_SMALL_TRAIN_FILE)
    validation_X_results,validation_Y_categories = readFileGetXY(Config.BSON_SMALL_VALIDATION_FILE)

    print("Train file : nb items %i nb categories %i" % (len(train_X_results),len(train_Y_categories)))
    print("Validation file : nb items %i nb categories %i" % (len(validation_X_results),len(validation_Y_categories)))
   
    model = LogisticRegression()
    print("start fit")
    model.fit(train_X_results,train_Y_categories)
    print("end fit")

    prediction = model.predict(validation_X_results)
    accuracy = get_accuracy(prediction, validation_Y_categories)

    print("accuracy = %f" % accuracy)

    model_parameters = model.get_params()
    save_model_parameters(model_parameters)
    
    t2 = time.time()
    
    total_time = t2 - t1
    print("Total time : %s" % str(total_time))
    print("end TestLogisticRegressionScikit")


runLogisticRegression()


