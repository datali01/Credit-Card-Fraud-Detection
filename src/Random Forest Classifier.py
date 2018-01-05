from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def isoForest(k):
    trainX = []
    testX = []
    trainY = []
    testY = []
    flag = True
    countTrain = 0
    print ("Fold " + str(k+1))
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            if countTrain >= 228000:                            #test on 20% of data
                break
            countTrain += 1
            if (k*57000) < countTrain < ((k+1)*57000):          #CV on 80% of data
                testX.append([float(i) for i in row[:-1]])
                testY.append(int(row[-1]))
            else:
                trainX.append([float(i) for i in row[:-1]])
                trainY.append(int(row[-1]))
    trainX = scalar.fit_transform(trainX)
    testX = scalar.fit_transform(testX)
    print ("Data Loaded")

    clf = RandomForestClassifier(max_features = 30)#random_state=rng
    #trainX = scalar.fit_transform(trainX)
    clf.fit(trainX, trainY)
    print ("Trained")
    predictY = clf.predict(testX)
    Y = predictY
    print ("Results")
    auc = roc_auc_score(testY, Y)
    print(auc)
    fpr,_,_ = roc_curve(testY, Y)
    print (fpr[1])
    _, recall, _ = precision_recall_curve(testY, Y)
    print (recall[1])
    return auc, fpr[1], recall[1]

#Main Function to do Cross validation followed by Testing
def main():
    auc = [0] * 4
    fpr = [0] * 4
    recall = [0] * 4

    print ("Running 4 fold CV on Isolation Forest.")
    for k in range(4):
        auc[k],fpr[k], recall[k] = isoForest(k)

    print ("Results averaged over 4 folds: ")
    print ("Area under curve : " + str(np.mean(auc)))
    print ("False Positive Rate : " + str(np.mean(fpr)))
    print ("Recall : " + str(np.mean(recall)))

    start_time = time.time()
    trainX = []
    testX = []
    trainY = []
    testY = []
    flag = True
    countTrain = 0

    print ("\n\nNow testing on separate data.")
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            countTrain += 1
            if countTrain > 228000:          #CV on 80% of data
                testX.append([float(i) for i in row[:-1]])
                testY.append(int(row[-1]))
            else:
                trainX.append([float(i) for i in row[:-1]])
                trainY.append(int(row[-1]))
    trainX = scalar.fit_transform(trainX)
    testX = scalar.fit_transform(testX)
    print ("Data Loaded")
    #trainX = scalar.fit_transform(trainX)
    clf = RandomForestClassifier(max_features = 30)  # random_state=rng
    clf.fit(trainX, trainY)
    print ("Trained")
    predictY = clf.predict(testX)
    Y = predictY
    print("%s seconds" % (time.time() - start_time))
    print ("Results")
    auc = roc_auc_score(testY, Y)
    print("Area under curve : " + str(auc))
    fpr,tpr, _ = roc_curve(testY, Y)
    print ("False Positive Rate : " + str(fpr[1]))
    _, recall, _ = precision_recall_curve(testY, Y)
    print ("Recall : "+str(recall[1]))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()