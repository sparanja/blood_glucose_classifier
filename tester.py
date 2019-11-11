import classifier as model
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

testFileName = ""

def cleanCGMData(filename):
    print("Inside cleanCGM Data")
    columnNames = []
    prevCGM = None
    for i in range(0, 31):
        columnNames.append('column'+str(i))
    finalDF = pd.DataFrame(columns = columnNames)
    cgmFrame = pd.read_csv(filename+'.csv', header=None, names = columnNames)
    for start, row in cgmFrame.iterrows():
        for end in range(0, 31):
            if(np.isnan(row[end]) or row[end] is None and prevCGM is not None):
                cgmFrame.iat[start, end] = prevCGM
            prevCGM = row[end]
    cgmFrame = cgmFrame.fillna(method='ffill')
    cgmFrame = cgmFrame.fillna(method='bfill')
    finalDF = pd.concat([finalDF, cgmFrame])
    return finalDF

def load_model_and_predict(modelName, data):
    print("Test results of classifier:"+modelName)
    with open(modelName, 'rb') as file:  
        classifier = pickle.load(file)
    global testFileName
    testFrame = pd.read_csv(testFileName+".csv")
    X = data.iloc[:, :-1].values
    y = testFrame['Meal']
    y_pred = classifier.predict(X)
    print("Prediction Vector:")
    print(y_pred)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

def main():
    print("Enter the name of the test csv file:")
    filename = input()
    cleanFrame = cleanCGMData(filename)
    cleanFrame.reset_index(drop=True, inplace=True)
    extractedFrame = model.extractFeatures(cleanFrame.transpose(), "")
    reducedFrame = model.reduceDimensions(extractedFrame)
    
    global testFileName
    print("Enter the test labels file name:")
    testFileName = input()
    
    load_model_and_predict("knnClassifier.pkl", reducedFrame)
    load_model_and_predict("neuralNetClassifier.pkl", reducedFrame)
    load_model_and_predict("desicisionTreeClassifier.pkl", reducedFrame)
    load_model_and_predict("randomForestClassifier.pkl", reducedFrame)




if __name__ == "__main__":
    main()