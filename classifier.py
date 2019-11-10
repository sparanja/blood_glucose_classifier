'''
Author: Sumanth Paranjape
Function: * Preprocesses data from meal and noMeal csv files, reduces number of dimensions 
          using PCA. 
          * Trains two models using k nearest neighbor and desicion tree classifiers
Libraries: pandas, numpy, scipy, sckit-learn
#New change
'''
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import iqr

def cleanCGMData(filename):
    print("Inside cleanCGM Data")
    columnNames = []
    prevCGM = None
    for i in range(0, 31):
        columnNames.append('column'+str(i))
    finalDF = pd.DataFrame(columns = columnNames)
    for j in range(1, 6):
        cgmFrame = pd.read_csv(filename+str(j)+'.csv', header=None, names = columnNames)
        for start, row in cgmFrame.iterrows():
            for end in range(0, 31):
                if(np.isnan(row[end]) or row[end] is None and prevCGM is not None):
                    cgmFrame.iat[start, end] = prevCGM
                prevCGM = row[end]
        cgmFrame = cgmFrame.fillna(method='ffill')
        cgmFrame = cgmFrame.fillna(method='bfill')
        finalDF = pd.concat([finalDF, cgmFrame])
    return finalDF

def getFeatureNames():
    return ['FFT', 'AverageCGMOnIntake', 'Hypoglycemia', 'Hyperglycemia', 'standard_deviation', 'range', 'Meal']
    
def extractFeatures(mergedFrame, mealLabel):
    ''' 
    'preMealAvg', 'postMealAvg','percentAbove300','percentBelow50','variance','Meal'
    '''
    #, 'Mean'
    featureMatrix = pd.DataFrame(columns=getFeatureNames())
    featureList = []
    fftArray = None
    for index in range(0, len(mergedFrame.columns)):
        cgmList = mergedFrame[index]
        featureList = list([np.var(fft(cgmList)),averageCGMFoodIntake(cgmList), percentHypoglycemia(cgmList),percentHyperglycemia(cgmList), np.std(cgmList), findrange(cgmList), mealLabel])
        features = {}
        i = 0
        for feature in getFeatureNames():
            features[feature] = featureList[i]
            i += 1            
        featureMatrix = featureMatrix.append(features, ignore_index=True)
    print(features)
    return featureMatrix

def averageCGMFoodIntake(cgmList):
    beforeMealAvg = np.mean(cgmList[0: 7])
    afterMealAvg = np.mean(cgmList[8: len(cgmList)])
    return abs(afterMealAvg-beforeMealAvg)

def percentHypoglycemia(cgmList):
    return (len(cgmList[cgmList<=70])/len(cgmList))*100

def percentHyperglycemia(cgmList):
    return (len(cgmList[cgmList>=180])/len(cgmList))*100

def percentAboveBG300(cgmList):
    return (len(cgmList[cgmList>300])/len(cgmList))*100

def percentBelowBG50(cgmList):
    return (len(cgmList[cgmList<150])/len(cgmList))*100

def preMealAverageBG(cgmList):
    lst = cgmList[:6]
    return sum(lst) / len(lst)

def postMealAverageBG(cgmList):
    lst = cgmList[7:30]
    return sum(lst) / len(lst)

def findrange(cgmList):
    highest = max(cgmList)
    lowest = min(cgmList)
    return highest-lowest

def reduceDimensions(featureMatrix):
    print(featureMatrix)
    mealLabelCol = featureMatrix['Meal']    
    featureMatrix = featureMatrix.drop(columns='Meal')
    #'Mean'
    features = getFeatureNames()
    normalize = featureMatrix.loc[:, features[0:len(features)-1]].values
    normalize = StandardScaler().fit_transform(normalize)
    #print(normalize)
    pca = PCA(n_components=5)
    principleComponents = pca.fit_transform(normalize)
    reducedDF = pd.DataFrame(data = principleComponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    reducedDF['Meal'] = mealLabelCol
    return reducedDF

def k_nearest_neighbor_classifier(featureMatrix):
    print("============K nearest neighbor classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    print('X value:')
    print(X)
    print('y value:')
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
     
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
def desicion_tree_classifier(featureMatrix):
    print("============Desicion Tree classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

    #Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
def neural_net_classifier(featureMatrix):
    print("=================Neural Net Classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    clf= MLPClassifier(hidden_layer_sizes=(3,), activation='logistic',
                       solver='adam', alpha=0.0001,learning_rate='constant', 
                      learning_rate_init=0.001)
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
    
    #Train Decision Tree Classifer
    mlp = mlp.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = mlp.predict(X_test)
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
def support_vector_machine(featureMatrix):
    print("=================Support Vector Machine Classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
def random_forest_classifier(featureMatrix):
    print("=================Random Forest Classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

def main():
    #Preprocesses CGM data and cleans missing values.
    cgmMealFrame = cleanCGMData("mealData")
    cgmNoMealFrame = cleanCGMData("Nomeal")
    cgmMealFrame.reset_index(drop=True, inplace=True)
    cgmNoMealFrame.reset_index(drop=True, inplace=True)
    
    print("Merged meal data:")
    print(cgmMealFrame)
    print("Merged no meal data:")
    print(cgmNoMealFrame)
    
    #Extract features from meal and no-meal data
    mealTrans = cgmMealFrame.transpose()
    print(mealTrans)
    mealFeatureMatrix = extractFeatures(mealTrans, "yes")
    noMealTrans = cgmNoMealFrame.transpose()
    print(noMealTrans)
    noMealFeatureMatrix = extractFeatures(noMealTrans, "no")
    
    #concatenate the two extracted feature matrix into one.
    combinedMatrix = pd.concat([mealFeatureMatrix, noMealFeatureMatrix])
    combinedMatrix.reset_index(drop=True, inplace=True)
    print("Combined feature matrix")
    print(combinedMatrix)
    
    #reduce the dimensionality using PCA.
    reducedMatrix = reduceDimensions(combinedMatrix)
    print("After dimensionality reduction:")
    print(reducedMatrix)
    
    #train the machine learning model
    #MODEL_1: K-Nearest-Neighbor classifier
    k_nearest_neighbor_classifier(reducedMatrix)
    
    #MODEL_2: Descision Tree Classifier
    desicion_tree_classifier(reducedMatrix)
    
    #MODEL_3: Neural Net Classifier
    neural_net_classifier(reducedMatrix)
    
    #MODEL_4: Support Vector Machine
    support_vector_machine(reducedMatrix)
    
    #MODEL_5: Random Forest Classifier
    random_forest_classifier(reducedMatrix)
    

if __name__ == "__main__":
    main()