import sys

if sys.argv[1] == "VM":
    BSON_TRAIN_FILE = "/datadrive/KaggleCdiscount/train.bson"
    ENVIRONMENT = "VM"
    MAX_PRODUCTS_TO_PROCESS = 100000
else:
    BSON_TRAIN_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
    ENVIRONMENT = "LAPTOP"
    MAX_PRODUCTS_TO_PROCESS = 1000
