import csv
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
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
def StrLOF(r):
    trainX = []
    testX = []
    flag = True
    testY = []
    countTrain = 0
    print ("Fold " + str(r + 1))
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            countTrain += 1
            if row[-1] == '1' or ((r*57000) < countTrain < ((r+1)*57000)):
                testX.append([float(i) for i in row[:-1]])
                testY.append(int(row[-1]))
            else:
                trainX.append([float(i) for i in row[:-1]])
	
    trainX = scalar.fit_transform(trainX)
    testX = scalar.fit_transform(testX)
    print ("Data Loaded")
    newX = trainX
    #newX = np.fft.fft(trainX)													#to remove the noise
    warnings.filterwarnings("ignore")
    clf = NearestNeighbors(n_neighbors=3, n_jobs=-1) 
    distances, _ = clf.fit(newX).kneighbors(newX)
    m = 500
    reachdist, reachindices = reachDist(newX, m, distances)
    irdMatrix = lrd(m, reachdist)
    lofScores = lof(irdMatrix, m, reachindices)
    alpha = lofScores
    alpha.sort(reverse=True)

    #newTestX = testX
    newTestX = np.fft.fft(testX)
    warnings.filterwarnings("ignore")
    distances, _ = clf.kneighbors(newTestX)
    reachdist, reachindices = reachDist(newTestX, m, distances)
    irdMatrix = lrd(m, reachdist)
    lofScores = lof(irdMatrix, m, reachindices)

    conf = 0.05

    Y = []
    num = 0.0
    for i in range(len(testX)):
        b = 0.0
        strangeness_i = lofScores[i]
        for j in range(len(alpha)):
            if strangeness_i > alpha[j]:
                break
            b += 1.0
        pvalue = (b+1.0)/(float(len(alpha))+1.0)
        if pvalue < conf:
            num += 1.0
            Y.append(1)
        else:
            Y.append(0)

    print ("Results")
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
    
    print ("Running 4 fold CV on StrOUD Algorithm using LOF as strangeness function.")
    for k in range(4):
        auc[k],fpr[k], recall[k] = StrLOF(k)

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
    newX = trainX
    # newX = np.fft.fft(trainX)
    warnings.filterwarnings("ignore")
    clf = NearestNeighbors(n_neighbors=3, n_jobs=-1)
    distances, _ = clf.fit(newX).kneighbors(newX)
    m = 15
    reachdist, reachindices = reachDist(newX, m, distances)
    irdMatrix = lrd(m, reachdist)
    lofScores = lof(irdMatrix, m, reachindices)
    alpha = lofScores
    alpha.sort(reverse=True)

    # newTestX = testX
    newTestX = np.fft.fft(testX)
    warnings.filterwarnings("ignore")
    distances, _ = clf.kneighbors(newTestX)
    reachdist, reachindices = reachDist(newTestX, m, distances)
    irdMatrix = lrd(m, reachdist)
    lofScores = lof(irdMatrix, m, reachindices)

    Y = []
    conf = 0.03								#best confidence value from CV
    # StrOUD Algorithm
    for i in range(len(testY)):
        b = 0.0
        strangeness_i = lofScores[i]
        for j in range(len(alpha)):
            if strangeness_i > alpha[j]:
                break
            b += 1.0
        pvalue = (b + 1.0) / (float(len(alpha)) + 1.0)
        if pvalue < conf:
            Y.append(1)
        else:
            Y.append(0)
			
    # prints running time
    print("%s seconds" % (time.time() - start_time))
	
	#prints results
    print ("Results")
    auc = roc_auc_score(testY, Y)
    print("Area under curve : " + str(auc))
    fpr, tpr, _ = roc_curve(testY, Y)
    print ("False Positive Rate : " + str(fpr[1]))
    _, recall, _ = precision_recall_curve(testY, Y)
    print ("Recall : " + str(recall[1]))

    #to plot an ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()