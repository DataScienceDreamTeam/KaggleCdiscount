import sys

if sys.argv[1] == "VM":
    # BSON_TRAIN_FILE = "/datadrive/KaggleCdiscount/train.bson"
    BSON_TRAIN_FILE = "/mnt/train.bson"
    ENVIRONMENT = "VM"
    MAX_PRODUCTS_TO_PROCESS = 20000000
    MULTI_PROCESS_BATCH_SIZE = 200000
    MULTI_PROCESS_NB_ITER = 36

elif sys.argv[1] == "DESKTOP":

    BSON_TRAIN_FILE = "G:\\KaggleCdiscount\\train.bson"
    SOURCE_CATEGORY_NAMES_CSV = "C:\\Users\\gaels\\Documents\\GitHub\\KaggleCdiscount\\Data\\category_names.csv"
    DEST_CATEGORY_NAMES_CSV = "C:\\Users\\gaels\\Documents\\GitHub\\KaggleCdiscount\\Data\\small_random_category_names.csv"

    BSON_SMALL_TRAIN_FILE = "C:\\Users\\gaels\\Documents\\GitHub\\KaggleCdiscount\\Data\\small_train.bson"
    BSON_SMALL_VALIDATION_FILE = "C:\\Users\\gaels\\Documents\\GitHub\\KaggleCdiscount\\Data\\small_validation.bson"

    NUMPY_RANDOM_SEED = 779493057
    NB_CATEGORIES_TO_KEEP = 3 # Total cat = 5270
    PERCENT_TRAIN = 0.8

    ENVIRONMENT = "DESKTOP"
    MAX_PRODUCTS_TO_PROCESS = 20000000
    MULTI_PROCESS_BATCH_SIZE = 200000
    MULTI_PROCESS_NB_ITER = 36
    WRITE_FILE_DIRECTORY = "G:\\KaggleCdiscount\\Temp"


    #Extraction 
    MAX_PICTURES_TO_EXTRACT = None

    
elif sys.argv[1] == "LAPTOP":

    BSON_TRAIN_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
    SOURCE_CATEGORY_NAMES_CSV = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\category_names.csv"
    DEST_CATEGORY_NAMES_CSV = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\small_random_category_names.csv"
    
    BSON_SMALL_TRAIN_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\small_train.bson"
    BSON_SMALL_VALIDATION_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\small_validation.bson"

    NUMPY_RANDOM_SEED = 779493057
    NB_CATEGORIES_TO_KEEP = 5 # Total cat = 5270
    PERCENT_TRAIN = 0.8


    ENVIRONMENT = "LAPTOP"
    MAX_PRODUCTS_TO_PROCESS = 20000000
    MULTI_PROCESS_BATCH_SIZE = 200000
    MULTI_PROCESS_NB_ITER = 36
    WRITE_FILE_DIRECTORY = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\Temp"

    #Extraction 
    MAX_PICTURES_TO_EXTRACT = 100