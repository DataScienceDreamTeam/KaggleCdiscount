import sys

if sys.argv[1] == "VM":
    # BSON_TRAIN_FILE = "/datadrive/KaggleCdiscount/train.bson"
    BSON_TRAIN_FILE = "/mnt/train.bson"
    ENVIRONMENT = "VM"
    MAX_PRODUCTS_TO_PROCESS = 20000000
    MULTI_PROCESS_BATCH_SIZE = 200000
    MULTI_PROCESS_NB_ITER = 36
elif sys.argv[1] == "DESKTOP":

    BSON_TRAIN_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
    ENVIRONMENT = "LAPTOP"
    MAX_PRODUCTS_TO_PROCESS = 20000000
    MULTI_PROCESS_BATCH_SIZE = 200000
    MULTI_PROCESS_NB_ITER = 36
    
elif sys.argv[1] == "LAPTOP":

    BSON_TRAIN_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
    ENVIRONMENT = "LAPTOP"
    MAX_PRODUCTS_TO_PROCESS = 20000000
    MULTI_PROCESS_BATCH_SIZE = 200000
    MULTI_PROCESS_NB_ITER = 36