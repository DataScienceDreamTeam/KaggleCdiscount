import numpy as np
import pandas as pd
import bson 
import random
import os
import io

from skimage.data import imread

class ProcessBson:

    def __init__(self, nbLinesPerFile, percentSplitTrain, filePathTrain, destFilePathDirectory,  nbItems = None):

        random.seed(0.7894709953495803)
        
        self.filePathTrain = filePathTrain
        self.destFilePathDirectory = destFilePathDirectory

        self.nbLinesPerFile = nbLinesPerFile #Power of 2 131072
        self.percentSplitTrain = percentSplitTrain
        self.nbItems = nbItems

    # Number of products : 7069896
    # Number of images : 12371293



    def generateCsvFiles(self, numberOfProducts):

        print("Start generateCsvFiles")

        np.set_printoptions(threshold=np.inf)

        train_iter = bson.decode_file_iter(open(self.filePathTrain, "rb"))

        countProduct = 0
        countImages = 0

        fileTrain = self.destFilePathDirectory + "\\" +"train.csv"
        fileValidation = self.destFilePathDirectory + "\\" +"validation.csv"

        with open(fileTrain, 'w') as newFileTrain:
            newFileTrain.write("product_id;category_id;img;img_index\n")
        with open(fileValidation, 'w') as newFileValidation:
            newFileValidation.write("product_id;category_id;img;img_index\n")

        df_train = pd.DataFrame(columns=('product_id', 'category_id', 'img', 'img_index'))
        df_validation = pd.DataFrame(columns=('product_id', 'category_id','img_index',  'img'))

        nbTrain = 0
        nbValidation = 0

        print("a")

        with open(fileTrain, 'a') as newFileTrain:
            with open(fileValidation, 'a') as newFileValidation:

                for c, d in enumerate(train_iter):
                    countProduct = countProduct + 1
                    print("b")
                    randNum = random.random()
                    isTrain = True
                    if(randNum > self.percentSplitTrain):
                        isTrain = False
        
                    for index_image, image in enumerate(d["imgs"]):
                        countImages = countImages + 1
                        picture = imread(io.BytesIO(image['picture']))

                        print("c")
                        if(isTrain == True):
                            print("h")
                            df_train.loc[nbTrain % 100] = [d['_id'],d['category_id'], picture, index_image ]

                            nbTrain = nbTrain + 1
                            if(nbTrain % 100 == 0):
                                print("d")
                                df_train.to_csv(newFileTrain, header=False, index=False)
                                df_train = pd.DataFrame(columns=('product_id', 'category_id', 'img', 'img_index'))

                        else:
                            print("i")
                            df_validation.loc[nbValidation % 100] = [d['_id'],d['category_id'], picture, index_image ]
                            nbValidation = nbValidation + 1

                            if(nbValidation % 100 == 0):
                                 print("e")
                                 df_validation.to_csv(newFileValidation, header=False, index=False)
                                 df_validation = pd.DataFrame(columns=('product_id', 'category_id', 'img', 'img_index'))


                    if(countProduct == numberOfProducts):
                        break
                if(nbTrain > 0):
                    df_train.to_csv(newFileTrain, header=False, index=False)
                if(nbValidation > 0):
                    df_validation.to_csv(newFileValidation, header=False, index=False)
                
        print("End generateCsvFiles")





    def getNumberOfProductsAndImages(self, productCountBreak = None):

        print("Start getNumberOfProductsAndImages")

        train_iter = bson.decode_file_iter(open(self.filePathTrain, 'rb'))

        countProduct = 0
        countImages = 0

        for c, d in enumerate(train_iter):
            countProduct = countProduct + 1

            for index_image, image in enumerate(d['imgs']):
                countImages = countImages + 1

            if(countProduct == productCountBreak):
                break

            if(countProduct % 500000 == 0):
                print("Number of products : %i" % countProduct)
                print("Number of images : %i" % countImages)

        print("End getNumberOfProductsAndImages")

        return countProduct, countImages

    # generate files with header, returns dictionary, filename + nblines
    def generateTrainCsvFiles(self):

        print("Start generateTrainCsvFiles")

        nbProducts, self.nbImagesInTrain = self.getNumberOfProductsAndImages(self.nbProductsInTrain)

        self.nbTrainFiles = self.nbImagesInTrain // self.nbLinesPerFile
        rest = self.nbImagesInTrain % self.nbLinesPerFile

        if(rest > 0):
            self.nbTrainFiles = self.nbTrainFiles + 1

        print("Nb Images In Train : %i" % self.nbImagesInTrain)
        print("Nb Lines Per File : %i" % self.nbLinesPerFile)
        print("Nb Files : %i" % self.nbTrainFiles)
        
        self.trainFiles = {}

        for i in range(1,self.nbTrainFiles + 1, 1):
            
            fileName = "train_"f'{i:03}'".csv"

            fullFilePath = self.destFilePathDirectory + "\\" + fileName
            self.trainFiles[fileName] = 0
            print("Generating file : %s" % fullFilePath)
            with open(fullFilePath, 'w') as newFile:
                newFile.write("Col1;Col2\n")


        print("End generateTrainCsvFiles")

    def distributeRandomTrainItemsInFile(self):

        print("Start distributeRandomTrainItemsInFile")

        train_iter = bson.decode_file_iter(open(self.filePathTrain, 'rb'))

        countImages = 0
        trainFilesKeys = list(self.trainFiles)

        print(self.trainFiles)
        print(trainFilesKeys)

        for c, d in enumerate(train_iter):
            
            productId = d['_id']
            for index_image, image in enumerate(d['imgs']):

                countImages = countImages + 1
                
                print(self.trainFiles)
                print(trainFilesKeys)

                randomFileKey = random.choice(trainFilesKeys)
                self.trainFiles[randomFileKey] = self.trainFiles[randomFileKey] + 1 #nb lines added in file

                with open(self.destFilePathDirectory + "\\" + randomFileKey, "a") as randomFile:
                    randomFile.write(str(productId) + ";" + str(index_image) + "\n")

                if(self.trainFiles[randomFileKey] == self.nbLinesPerFile):
                    trainFilesKeys.remove(randomFileKey)

            if(countImages == self.nbImagesInTrain):
                break

        if(len(trainFilesKeys) == 0):
            print("All files are complete, nothing to do")
        # take the file with less items and complete the other incomplete files
        if(len(trainFilesKeys) == 1):
            print("All files are complete except one, swap name with the file with the highest id so the last file is the incomplete one")

            
            self.swapWithLastFileName(trainFilesKeys[0])

        if(len(trainFilesKeys) > 1):
            print("More than one incomplete file, take the file with less items and use it complete the other incomplete files. Then swap name of the incomplete file with the file with the highest id")

            minFileName = min(self.trainFiles, key = self.trainFiles.get)

            with open(self.destFilePathDirectory + "\\" +minFileName,"r") as minFile:
                contentMinFile = minFile.readlines()

            lastIndex = 1
            for incompleteTrainFileName in trainFilesKeys:
                if(incompleteTrainFileName != minFileName):
                    with open(self.destFilePathDirectory + "\\" + incompleteTrainFileName, "a") as incompleteFile:
                        while self.trainFiles[incompleteTrainFileName] != self.nbLinesPerFile:
                            lineMinFile = contentMinFile[lastIndex]
                            incompleteFile.write(lineMinFile)
                            self.trainFiles[incompleteTrainFileName] = self.trainFiles[incompleteTrainFileName] + 1
                            lastIndex = lastIndex + 1

            with open(self.destFilePathDirectory + "\\" +minFileName,"w") as minFile:
                for index, row in enumerate(contentMinFile):
                    if(index >= lastIndex):
                        minFile.write(row)

            self.swapWithLastFileName(minFileName)

        print("End distributeRandomTrainItemsInFile")

    def swapWithLastFileName(self, fileName):

            print("Start swapWithLastFileName")
            lastFileName = list(self.trainFiles)[-1]

            print("File name : %s" % fileName)
            print("Last file name : %s" % lastFileName)
            tempName = lastFileName + ".temp"

            os.rename(self.destFilePathDirectory + "\\" +lastFileName,self.destFilePathDirectory + "\\" +tempName)
            os.rename(self.destFilePathDirectory + "\\" +fileName,self.destFilePathDirectory + "\\" +lastFileName)
            os.rename(self.destFilePathDirectory + "\\" +tempName,self.destFilePathDirectory + "\\" +fileName)

            print("End swapWithLastFileName")

    # append lines to a selected random files in file dictionary
    # if nbLines in file = self.nbLinesPerFile remove file from dictionary
    def appendLineRandom(self, productId, indexImage):


        return

    def randomizeProducts(self):
        return

    def generateListOfProductImagesInFiles(self):
        return

    def generateFilesFromListOfProductImages(self):
        return
    
    def run(self):

        print("Start run")

        # if(self.nbItems == None):
        #     self.nbProducts, self.nbImages = self.getNumberOfProductsAndImages()
        # else:
        #     self.nbProducts, self.nbImages = self.getNumberOfProductsAndImages(self.nbItems)


        # self.nbProductsInTrain = int(self.nbProducts * self.percentSplitTrain)
        # self.nbProductsInValidation = self.nbProducts - self.nbProductsInTrain

        # print("Number of products : %i" % self.nbProducts)
        # print("Number of images : %i" % self.nbImages)
        # print("Number of nbProductsInTrain : %i" % self.nbProductsInTrain)
        # print("Number of nbProductsInValidation : %i" % self.nbProductsInValidation)

        # self.generateTrainCsvFiles()
        # self.distributeRandomTrainItemsInFile()

        self.generateCsvFiles(10)
        print("End run")