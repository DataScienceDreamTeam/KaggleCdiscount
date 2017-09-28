import sys

if sys.argv[0] == "VM":
    BSON_TRAIN_FILE = "/datadrive/KaggleCdiscount/train.bson"
    ENVIRONMENT = "VM"
    MAX_PRODUCTS_TO_PROCESS = None
else:
    BSON_TRAIN_FILE = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
    ENVIRONMENT = "LAPTOP"
    MAX_PRODUCTS_TO_PROCESS = 100000
