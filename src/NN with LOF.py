import csv
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#reachDist() calculates the reach distance of each point to MinPts around it
def reachDist(df, MinPts, knnDist):
    clf = NearestNeighbors(n_neighbors=MinPts)
    clf.fit(df)
    distancesMinPts, indicesMinPts = clf.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts

#lrd calculates the Local Reachability Density
def lrd(MinPts,knnDistMinPts):
    return (MinPts/np.sum(knnDistMinPts,axis=1))

#Finally lof calculates LOF outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof

#For Cross Validation
def LOF():
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
    #newTotalX = np.fft.fft(totalX)
    newTotalX = totalX
    warnings.filterwarnings("ignore")
    clf = NearestNeighbors(n_neighbors=5,  n_jobs = -1)
    clf.fit(newTotalX)
    distances, _ = clf.kneighbors(newTotalX)
    m = 500
    reachdist, reachindices = reachDist(newTotalX, m, distances)
    irdMatrix = lrd(m, reachdist)
    lofScores = lof(irdMatrix, m, reachindices)
    Y = []
    for i in range(len(totalY)):
        if lofScores[i] > 1.44:
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
    print ("Running 4 fold CV on LOF based Anomaly detection.")
    LOF()

    start_time = time.time()
    totalX = []
    totalY = []
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
                totalX.append([float(i) for i in row[:-1]])
                totalY.append(int(row[-1]))
    totalX = scalar.fit_transform(totalX)
    print ("Data Loaded")

    newTotalX = np.fft.fft(totalX)
    #newTotalX = totalX
    warnings.filterwarnings("ignore")
    clf = NearestNeighbors(n_neighbors=10,  n_jobs = -1)
    clf.fit(newTotalX)
    distances, _ = clf.kneighbors(newTotalX)
    m = 500
    reachdist, reachindices = reachDist(newTotalX, m, distances)
    irdMatrix = lrd(m, reachdist)
    lofScores = lof(irdMatrix, m, reachindices)
    Y = []
    for i in range(len(totalY)):
        if lofScores[i] > 1.44:
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
