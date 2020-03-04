import pandas as pd
import math
import csv

class Normalizar:
    rawData = None
    data = []
    inputFile = ""
    outputFileName = ""
    min = math.inf
    max = -math.inf

    def __init__(self) -> None:
        super().__init__()


    #read csv
    def getData(self):
        dataCSV = pd.read_csv(self.inputFile, delimiter=",", header=None)
        self.rawData = dataCSV.copy()
        for row in dataCSV.values:
            for col in row:
                if not math.isnan(col):
                    self.data.append(col)
        return

    def getParams(self):
        self.min = min(self.data)
        self.max = max(self.data)
        return

    def getFinalData(self):
        rowToAppend = []
        for row in self.rawData.values:
            for col in row:
                if not math.isnan(col):
                    normal:float = (col-self.min)/(self.max-self.min)
                    normal = float("{0:.6f}".format(normal))
                    rowToAppend.append(normal)
            self.writeToCSV(rowToAppend)
            rowToAppend = []
        return

    def writeToCSV(self, row:list):
        with open(self.outputFileName, 'a', newline="") as output:
            writer = csv.writer(output, delimiter=" ")
            writer.writerow(row)
        return

    def normalizar(self,fileName):
        self.inputFile = fileName
        self.outputFileName = "normalizado_" + self.inputFile
        open(self.outputFileName,"w+")
        self.getData()
        self.getParams()
        self.getFinalData()
        return
