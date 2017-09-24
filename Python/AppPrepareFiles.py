import ReadBson


processBson = ReadBson.ProcessBson(
    nbLinesPerFile = 128,
    percentSplitTrain= 0.8,
    nbItems = 1000,
    filePathTrain = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson",
    destFilePathDirectory = "E:\\KaggleCdiscountTrainSplit")

processBson.run()