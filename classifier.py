'''
Authors: Sumanth Paranjape
Function: * Preprocesses data from meal and noMeal csv files, reduces number of dimensions 
          using PCA. 
          * Trains two models using k nearest neighbor and desicion tree classifiers
Libraries: pandas, numpy, scipy, sckit-learn
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
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def cleanCGMData(filename):
    print("Preprocessing Data")
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
    return ['FFT', 'AverageCGMOnIntake', 'Hypoglycemia', 'Hyperglycemia', 'range', 'averageBeforeMeal', 'averageAfterMeal', 'Meal']
    
def extractFeatures(mergedFrame, mealLabel):
    #, 'Mean'
    featureMatrix = pd.DataFrame(columns=getFeatureNames())
    featureList = []
    fftArray = None
    for index in range(0, len(mergedFrame.columns)):
        cgmList = mergedFrame[index]
        
        featureList = list([np.var(fft(cgmList)),averageCGMFoodIntake(cgmList), percentHypoglycemia(cgmList),percentHyperglycemia(cgmList), findrange(cgmList), averageBeforeMeal(cgmList),averageAfterMeal(cgmList)])
        featureList.append(mealLabel)
        
        features = {}
        i = 0
        for feature in getFeatureNames():
            features[feature] = featureList[i]
            i += 1
        featureMatrix = featureMatrix.append(features, ignore_index=True)
    return featureMatrix

def averageCGMFoodIntake(cgmList):
    beforeMealAvg = np.mean(cgmList[0: 7])
    afterMealAvg = np.mean(cgmList[13: len(cgmList)])
    return abs(afterMealAvg-beforeMealAvg)

def averageBeforeMeal(cgmList):
    return np.mean(cgmList[0: 7])

def averageAfterMeal(cgmList):
    return np.mean(cgmList[13: len(cgmList)])

def percentHypoglycemia(cgmList):
    return (len(cgmList[cgmList<=70])/len(cgmList))*100

def percentHyperglycemia(cgmList):
    return (len(cgmList[cgmList>=180])/len(cgmList))*100

def findrange(cgmList):
    highest = max(cgmList)
    lowest = min(cgmList)
    return highest-lowest

def stdBeforeMeal(cgmList):
    return np.var(cgmList[:7])

def stdAfterMeal(cgmList):
    return np.var(cgmList[13:30])

def reduceDimensions(featureMatrix):
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

def save_model(name, model):
    model_path = name+".pkl"  

    with open(model_path, 'wb') as file:  
        pickle.dump(model, file)

def k_nearest_neighbor_classifier(featureMatrix):
    print("============K nearest neighbor classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
         
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy of k-fold cross validation")
    print(k_fold_cross_validation(classifier, X, y))
    save_model("kNNClassifier", classifier)
    
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
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy of k-fold cross validation")
    print(k_fold_cross_validation(clf, X, y))
    save_model("desicisionTreeClassifier", clf)
    
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
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy of k-fold cross validation")
    print(k_fold_cross_validation(mlp, X, y))
    save_model("neuralNetClassifier", mlp)
    
def support_vector_machine(featureMatrix):
    print("=================Support Vector Machine Classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy of k-fold cross validation")
    print(k_fold_cross_validation(svclassifier, X, y))
    save_model("svmClassifier", svclassifier)
    
def random_forest_classifier(featureMatrix):
    print("=================Random Forest Classifier====================")
    X = featureMatrix.iloc[:, :-1].values
    y = featureMatrix.iloc[:, 5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    clf=RandomForestClassifier(n_estimators=300)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy of k-fold cross validation")
    print(k_fold_cross_validation(clf, X, y))
    save_model("randomForestClassifier", clf)

def k_fold_cross_validation(model, X, y):
    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    accuracies = []

    for train_index, test_index in kf.split(X):
        data_train   = X[train_index]
        target_train = y[train_index]
        data_test    = X[test_index]
        target_test  = y[test_index]
        model.fit(data_train,target_train)
        preds = model.predict(data_test)
        accuracy = accuracy_score(target_test,preds)
        accuracies.append(accuracy)
    return np.max(accuracies)

def main():
    #Preprocesses CGM data and cleans missing values.
    cgmMealFrame = cleanCGMData("mealData")
    cgmNoMealFrame = cleanCGMData("Nomeal")
    cgmMealFrame.reset_index(drop=True, inplace=True)
    cgmNoMealFrame.reset_index(drop=True, inplace=True)
    
    #Extract features from meal and no-meal data
    mealTrans = cgmMealFrame.transpose()
    mealFeatureMatrix = extractFeatures(mealTrans, "yes")
    noMealTrans = cgmNoMealFrame.transpose()
    noMealFeatureMatrix = extractFeatures(noMealTrans, "no")
    
    #concatenate the two extracted feature matrix into one.
    combinedMatrix = pd.concat([mealFeatureMatrix, noMealFeatureMatrix])
    combinedMatrix.reset_index(drop=True, inplace=True)
    
    #reduce the dimensionality using PCA.
    reducedMatrix = reduceDimensions(combinedMatrix)
    
    #train the machine learning model
    print("Training begins..")
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