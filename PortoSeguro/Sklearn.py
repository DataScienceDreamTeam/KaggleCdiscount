# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:12:52 2017

Using SKLEARN

@author: mdarq
"""

%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time

class encoding:
    """
    This will apply sklearn.preprocessing.LabelBinarizer and sklearn.preprocessing.StandardScaler
    :example:
    >>> Test
    Blabla
    >>> Other test
    TTT blabla
    
    :param data: A pandas DataFrame

    """

    def __init__(self):
        #binary, categorical or numerical features
        self.binary = list()
        self.categorical = list()
        self.numeric = list()
        self.feat_remove = list()
#        self.data = dat

    def __repr__(self):
        return "<Binarizer and Scaler> object"
    
    def get_cat(self,dat):
        colnames = dat.columns
        print(colnames)
        for col in colnames:
            c = col.split('_')[-1]
            if c == "cat":
                self.categorical.append(col)                                
            elif c == "bin":
                self.binary.append(col)             
            elif (c != "cat")&(c != "bin"):
                self.numeric.append(col)
        return self.categorical, self.binary, self.numeric
    
    def NA_encode(self,dat):
        #Implement 'most frequent' strategy
        print("Processing categorical features...")
        for c in self.categorical:
            #Null values are represented by "-1", switching to NaN
            print("Number of missing values for feature: {0}".format(c))
            dat.loc[(dat[c] == -1, c)] = np.NaN
            na_count = dat[c].isnull().sum()
            dat[c].fillna(value=dat[c].mode()[0], inplace=True)
            print(na_count) #Checking number of null values            
        #Implement 'mean' or 'median" strategy
        print("Processing numerical features...")
        for c in self.numeric:
            #NaN are represented as "-1" in the original data
            dat.loc[(dat[c] == -1, c)] = np.NaN
            na_count = dat[c].isnull().sum()
            print("Number of missing values for feature: {0}\n{1}".format(c,na_count))
            #Replace NaN values
            dat[c].fillna(value=dat[c].mean(), inplace=True)

    def Binarize_scale(self, dat):
        """
        Get column names and determine if numeric/cat/bin
        :param data: A pandas DataFrame
        """
        self.get_cat(dat)
        print("Encoding null values")
        self.NA_encode(dat)
        print("Binarizing...")
#       Let's first create the LabelBinarized features    
        '''          
        label = LabelBinarizer()
        for c in self.categorical:   
            _ = label.fit_transform(dat[c])
            for i in range(np.shape(_)[1]):              
#               self.data[str(c)+"_"+str(i)] = _[:,i]
               dat[str(c)+"_"+str(i)] = _[:,i]
        print("Scaling...")
#       Scale numeric features
        scaler = StandardScaler()
        for c in self.numeric:
            _ = scaler.fit_transform(np.float64(dat[c]).reshape(-1,1))
            '''  
        pass  
        return dat

def RFmodel(n_estimators=10,min_samples_leaf=10):
    '''
    Defining a random forest classifier, Fitting on training set, Prediction on test set
    ---
    Returns    
    train_score : the accuracy score calculated on the training set
    test_score : the accuracy score calculated on the test set
    auc_roc : auc_roc estimated on test set
    RF_model : the Random Forest model for further prediction
    '''
    RF_model = RandomForestClassifier(n_jobs=-1,n_estimators=150,oob_score=True,min_samples_leaf=min_samples_leaf)
    t_init = time.time()
    print("Fitting with \n n_estimators = {0} \n min_samples_leaf = {1} ".format(n_estimators, min_samples_leaf))
    RF_model.fit(X_train,y_train)
    t_final = time.time()
    print("Total time: {0}s".format(t_final-t_init))
    train_score = RF_model.score(X_train,y_train)
    test_score = RF_model.score(X_test,y_test)
    print("Train score: {0}".format(train_score))
    print("Test score: {0}".format(test_score))
    pred_RF_test = RF_model.predict(X_test)
    print(pred_RF_test.sum())
    auc_roc = roc_auc_score(y_test,pred_RF_test)
    print("AUC-ROC: {0}\n".format(auc_roc))    
    return train_score, test_score, auc_roc, RF_model

def logitmodel(C=1):
    """
    Trains and fits a logistic regression model
    """
    print("Fitting with \n C = {0}".format(C))
    logit_model = LogisticRegression(n_jobs=-1)
    t_init = time.time()
    logit_model.fit(X_train, y_train)
    t_final = time.time()
    print("Total time: {0}s".format(t_final-t_init))
    train_score = logit_model.score(X_train,y_train)
    test_score = logit_model.score(X_test,y_test)
    print(train_score)
    print(test_score)
    pred_logit_test = logit_model.predict(X_test)
    print(pred_logit_test.sum())
    auc_roc = roc_auc_score(y_test,pred_logit_test)
    print("AUC-ROC: {0}\n".format(auc_roc))  
    return train_score, test_score, auc_roc, logit_model

def ignorelargenan(dat):
    """
    Will delete columns if the number of NaN values is more than 20% of data
    """
    for c in dat.columns:
        dat.loc[(dat[c] == -1.0, c)] = np.NaN
        na_count = dat[c].isnull().sum()
        if (na_count > (len(data))/5):
            dat = dat.drop(columns=c)
    return dat
#%% OPEN FILES
data = pd.read_csv("./train.csv")
data_valid = pd.read_csv("test.csv")
#%% BINARIZE AND ENCODE
#Remove column if too many NaN values
data = ignorelargenan(data)

b = encoding()
data2 = b.Binarize_scale(dat=data)

c = encoding()
data_valid2 = c.Binarize_scale(dat=data_valid)

print("Data contains {0} categorical, {1} numerical and {2} binary features".format(len(b.categorical), len(b.numeric), len(b.binary)))
print("Validation data contains {0} categorical, {1} numerical and {2} binary features".format(len(c.categorical), len(c.numeric), len(c.binary)))

#%% FEATURES ARRAY

feat = list(data.columns)
feats = list(feat)

for f in feats:
    if (f.split('_')[-1]) == 'cat':
        feat.remove(str(f))

features = np.array(feat)
target = "target"
# Remove first column and target
features = np.delete(features,[0,1])
train, test = train_test_split(data, train_size=0.80)

#%% DEALING WITH DATA IMBALANCE
print("There are {:.2f}% of positive outcomes in the train set".format(100*data['target'].sum()/len(data)))

#Fetch only positive entries
df_positives_train_data = train.loc[train['target']==1]

#Duplicate to balance data
train = train.append(df_positives_train_data)
print("There are {:.2f}% of positive outcomes in the train set".format(100*train['target'].sum()/len(train)))
#%% CREATING TRAIN/TEST SETS
X_train, y_train, X_test, y_test = train[features], train[target], test[features], test[target]

#%% LOGIT MODEL
logit_model = LogisticRegression(n_jobs=-1,max_iter=60)
t_init = time.time()
logit_model.fit(X_train, y_train)
t_final = time.time()
print("Total time: {0}s".format(t_final-t_init))
#%%
print(logit_model.score(X_train,y_train))
print(logit_model.score(X_test,y_test))
pred_logit_test = logit_model.predict(X_test)
pred_logit_test_proba = logit_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, pred_logit_test_proba, pos_label=1)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
print("roc_auc score: {0}".format(roc_auc_score(y_test,pred_logit_test_proba)))
print("Somme de positifs sur test: {0}".format(y_test.sum()))
print("Somme de positifs sur la prédiction via logistic regression sur test set: {0}".format(pred_logit_test.sum()))
print("There are {:.2f}% of positive outcomes on the prediction test".format(100*pred_logit_test.sum()/len(test)))

#pred_logit_valid = logit_model.predict(data_valid2[features])
#print("This model predicts {0:.2f}% of positive results on validation set".format(100*pred_logit_valid.sum()/len(data_valid2)))


#%% LOGIT HYPERPARAMETERS TUNING
c=[0.00001, 0.00002, 0.00003, 0.00004, 0.00005,0.00007, 0.0001, 0.0005,0.001,0.005, 0.01,0.05,0.1]
train_score = []
test_score = []
auc_roc = []
for C in c:
    t1, t2, t3,logit_model = logitmodel(C=C)
    train_score.append(t1)
    test_score.append(t2)
    auc_roc.append(t3)
    
#%% RF HYPERPARAMETERS TUNING
''' min_samples_leaf for random Forest '''
msl = [1, 5, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100, 200,300,500]
train_score = []
test_score = []
auc_roc = []
for m in msl:
    t1,t2,t3 = RFmodel(min_samples_leaf=m)
    train_score.append(t1)
    test_score.append(t2)
    auc_roc.append(t3)
#%% RF PREDICT
train_score, test_score, auc_roc, RF_model = RFmodel(min_samples_leaf=120)

print("Predicting on test data...")
print("The Random Forest model yields : \nTrain score = {0}\nTest score = {1}\nAUC_ROC = {2}".format(train_score,test_score, auc_roc))
pred_RF_test_proba = RF_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, pred_RF_test_proba)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
plt.plot(fpr,tpr)
print("roc_auc score: {0}".format(roc_auc_score(y_test,pred_RF_test_proba)))

print("Predicting on validation data ...")
pred_RF_valid = RF_model.predict(data_valid2[features])
print("This model predicts {0:.2f}% of positive results on the validation data".format(100*pred_RF_valid.sum()/len(data_valid2)))

#%% TENSORFLOW

#train, test = train_test_split(data, train_size=0.90)
#X_train, y_train, X_test, y_test = train[features], train[target], test[features], test[target]


def model(hu, model_dir, features):
    # Specify the shape of the features columns
    feature_columns = [tf.feature_column.numeric_column("x", shape=[len(features),1])]
    # Build n layer DNN with hu units (hu is an array)
    # The default optimizer is "AdaGrad" but we can specify another model
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=hu,
                                        n_classes=2,    
                                        optimizer=tf.train.AdamOptimizer(
                                                learning_rate=0.1,
                                                beta1=0.8,
                                                beta2=0.99),
                                        model_dir=model_dir)
# Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_train)},
        y=np.array(y_train.values.reshape((len(y_train),1))),
        num_epochs=None,
        shuffle=True)
    return classifier, train_input_fn

# 3-layers
classifier, train_input_fn = model([210,400,600,400,200,100,50,2], "./tmp/DNN9", features)
#Let's train
classifier.train(input_fn=train_input_fn, steps=5000)

# Define the test inputs
def testinput(X_test, y_test):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(X_test)},
            y=np.array(y_test),
            num_epochs=1,
            shuffle=False)
    return test_input_fn
  
test_input_fn = testinput(X_test,y_test)      
pred_tf_test_data_temp = classifier.predict(input_fn=test_input_fn)
# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


my_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(data_valid2[features])},
        y=None,
        num_epochs=1,
        shuffle=False)

pred = classifier.predict(input_fn=my_input_fn)

predictions = list(pred)
pred_tf = list()

for i,p in enumerate(predictions):
    pred_tf.append(p['class_ids'][0])

pred_tf = np.array(pred_tf)
print(pred_tf.sum())

pred_tf = pred_tf.astype(int)
#%% EVALUATION OF NEURAL NETWORK CLASSIFICATION

pred_tf_test_data_temp = list(pred_tf_test_data_temp)
pred_tf_test_data_proba = list()
pred_tf_test_data = list()

for i,p in enumerate(pred_tf_test_data_temp):
    pred_tf_test_data_proba.append(p['probabilities'][1])
    pred_tf_test_data.append(p['class_ids'][0])
   
fpr, tpr, _ = roc_curve(y_test, pred_tf_test_data_proba)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
plt.plot(fpr,tpr)
print("roc_auc score: {0}".format(roc_auc_score(y_test,pred_tf_test_data)))
print(np.array(pred_tf_test_data).sum())
print(y_test.sum())
#%%Submission
#pred_avg = np.round((pred_logit_valid+pred_RF_valid+pred_tf)/3)
df = pd.DataFrame()
df["id"] = np.int32(data_valid2["id"])
df["target"] = np.int32(pred_logit_valid)

df.to_csv("pred9.csv", index=None, sep=",")

#%%
#%%
print("There are {:.2f}% of positive outcomes".format(100*data['target'].sum()/len(data)))

#Fetch only positive entries
df_positives_data = data.loc[data['target']==1]

#Duplicate to balance data
data = data.append(df_positives_data)
print("There are now {:.2f}% of positive outcomes".format(100*data['target'].sum()/len(data)))

#pred_RF = moisie
#pred_logit = moisie

#%%
