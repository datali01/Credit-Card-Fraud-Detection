import numpy as np
import csv
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def mahanalobis():
    totalX = []
    totalY = []
    flag = True
    countTrain = 0
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            if countTrain >= 228000:                            #test on 20% of data
                break
            countTrain += 1
            totalX.append([float(i) for i in row[:-1]])
            totalY.append(int(row[-1]))
    totalX = scalar.fit_transform(totalX)
    print ("Data Loaded")
    clf = EmpiricalCovariance()
    clf.fit(totalX)
    distances = clf.mahalanobis(totalX)

    Y = []
    for i in range(len(totalY)):

        if np.log10(distances[i]) > 1.838:
            Y.append(1)
        else:
            Y.append(0)
    print ("Results")
    auc = roc_auc_score(totalY, Y)
    print(auc)
    fpr, _, _ = roc_curve(totalY, Y)
    print (fpr[1])
    _, recall, _ = precision_recall_curve(totalY, Y)
    print (recall[1])
    return auc, fpr[1], recall[1]

#Main Function to do Cross validation followed by Testing
def main():
    print ("Running CV on Mahalanobis Distance based approach.")
    mahanalobis()

    start_time = time.time()
    totalX = []
    totalY = []
    flag = True
    countTrain = 228000
    print ("\n\nNow testing on separate data.")
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            countTrain += 1
            if countTrain > 228000:          #CV on 80% of data
                totalX.append([float(i) for i in row[:-1]])
                totalY.append(int(row[-1]))
    print ("Data Loaded")
    totalX = scalar.fit_transform(totalX)
    clf = EmpiricalCovariance()
    clf.fit(totalX)
    distances = clf.mahalanobis(totalX)

    Y = []
    for i in range(len(totalY)):

        if np.log10(distances[i]) > 1.838:
            Y.append(1)
        else:
            Y.append(0)
    print("%s seconds" % (time.time() - start_time))
    print ("Results")
    auc = roc_auc_score(totalY, Y)
    print("Area under curve : " + str(auc))
    fpr, tpr, _ = roc_curve(totalY, Y)
    print ("False Positive Rate : " + str(fpr[1]))
    _, recall, _ = precision_recall_curve(totalY, Y)
    print ("Recall : " + str(recall[1]))

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()