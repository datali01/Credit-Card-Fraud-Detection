import csv
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def StrKNN(r):
    trainX = []
    testX = []
    trainY = []
    testY = []
    flag = True
    countTrain = 0
    print ("Fold " + str(r + 1))
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            countTrain += 1
            if countTrain > 228000:
                break
            if row[-1] == '1' or (r*57000) < countTrain < ((r+1)*57000):
                testX.append([float(i) for i in row[:-1]])
                testY.append(int(row[-1]))
            else:
                trainX.append([float(i) for i in row[:-1]])
                trainY.append(int(row[-1]))
	trainX = scalar.fit_transform(trainX)
    testX = scalar.fit_transform(testX)
    print ("Data Loaded")
    newX = np.fft.fft(trainX)
    warnings.filterwarnings("ignore")
    clf = NearestNeighbors(n_neighbors=6, n_jobs=-1)
    clf.fit(newX)
    distances, _ = clf.kneighbors(newX)

    alpha = [0]*len(distances)
    #Calculating KNN scores from KNN distances
    for i in range(len(distances)):
        alpha[i] = sum(distances[i])

    alpha.sort(reverse=True)

    newTestX = np.fft.fft(testX)
    warnings.filterwarnings("ignore")
    #Applying KNN on new data set
    distances, _ = clf.kneighbors(newTestX)

    conf = 0.05

    Y = []
    num = 0.0
    for i in range(len(newTestX)):
        b = 0.0
        strangeness_i = sum(distances[i])
        for j in range(len(alpha)):
            if strangeness_i > alpha[j]:
                break
            b += 1.0
        pvalue = (b+1.0)/(float(len(newTestX))+1.0)
        if pvalue < conf:
            num += 1.0
            Y.append(1)
        else:
            Y.append(0)
    auc = roc_auc_score(testY, Y)
    print(auc)
    fpr, _, _ = roc_curve(testY, Y)
    print (fpr[1])
    _, recall, _ = precision_recall_curve(testY, Y)
    print (recall[1])
    return auc, fpr[1], recall[1]

#Main Function to do Cross validation followed by Testing
def main():
    auc = [0] * 4
    fpr = [0] * 4
    recall = [0] * 4
    
    print ("Running 4 fold CV on StrOUD Algorithm using KNN as strangeness function.")
    for k in range(4):
        auc[k],fpr[k], recall[k] = StrKNN(k)

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
            if countTrain > 228000 or row[-1] == '1':          #CV on 80% of data
                testX.append([float(i) for i in row[:-1]])
                testY.append(int(row[-1]))
            else:
                trainX.append([float(i) for i in row[:-1]])
                trainY.append(int(row[-1]))
    trainX = scalar.fit_transform(trainX)
    testX = scalar.fit_transform(testX)
    print ("Data Loaded")
    newX = np.fft.fft(trainX)
    warnings.filterwarnings("ignore")
    clf = NearestNeighbors(n_neighbors=6,n_jobs=-1)
    clf.fit(newX)
    distances, _ = clf.kneighbors(newX)

    alpha = [0] * len(distances)
    # Calculating KNN scores from KNN distances
    for i in range(len(distances)):
        alpha[i] = sum(distances[i])

    alpha.sort(reverse=True)

    newTestX = np.fft.fft(testX)
    warnings.filterwarnings("ignore")
    # Applying KNN on new data set
    distances, _ = clf.kneighbors(newTestX)

    conf = 0.09
    Y = []

    #StrOUD Algorithm
    for i in range(len(newTestX)):
        b = 0.0
        strangeness_i = sum(distances[i])
        for j in range(len(alpha)):
            if strangeness_i > alpha[j]:
                break
            b += 1.0
        pvalue = (b + 1.0) / (float(len(newTestX)) + 1.0)
        if pvalue < conf:
            Y.append(1)
        else:
            Y.append(0)
    #prints run time of algorithm
    print("%s seconds" % (time.time() - start_time))
    #prints results
    print ("Results")
    auc = roc_auc_score(testY, Y)
    print("Area under curve : " + str(auc))
    fpr, tpr, _ = roc_curve(testY, Y)
    print ("False Positive Rate : " + str(fpr[1]))
    _, recall, _ = precision_recall_curve(testY, Y)
    print ("Recall : " + str(recall[1]))

    #to plot ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr , color='darkorange', label='ROC curve (area = %0.3f)' % max_area)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()