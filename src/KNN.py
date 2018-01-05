import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def KNN():
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
            if countTrain >= 228000:								#doesn't use the last 20% of data
                break
            countTrain += 1
            totalX.append([float(i) for i in row[:-1]])
            totalY.append(int(row[-1]))
    totalX = scalar.fit_transform(totalX)
    print ("Data Loaded")
    #newX = np.fft.fft(totalX)
    warnings.filterwarnings("ignore")
    newX = totalX
    clf = NearestNeighbors(n_neighbors=6, n_jobs = -1)
    clf.fit(newX)
    distances, _ = clf.kneighbors(newX)
    Y = []
    length = len(totalY)

    # if distance from Kth nearest neighbor is 4 or more then the point is an outlier (less than 4 is an inlier)
	for i in range(length):
        distances[i].sort()
        if distances[i][-1] < 4:
            Y.append(0)
        else:
            Y.append(1)

    print ("Results")
    auc = roc_auc_score(totalY, Y)
    print(auc)
    fpr, _, _ = roc_curve(totalY, Y)
    print (fpr[1])
    _, recall, _ = precision_recall_curve(totalY, Y)
    print (recall[1])

#Main Function to do Cross validation followed by Testing
def main():
    print ("Running CV on KNN based Anomaly detection.")
    KNN()

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

    #newX = np.fft.fft(totalX)
    warnings.filterwarnings("ignore")
    newX = totalX
    clf = NearestNeighbors(n_neighbors=6, n_jobs=-1)
    clf.fit(newX)
    distances, _ = clf.kneighbors(newX)
    Y = []
    length = len(totalY)
    alpha = [0] * length
    # if distance from Kth nearest neighbor is 4 or more then the point is an outlier (less than 4 is an inlier)
    for i in range(length):
        distances[i].sort()
        if distances[i][-1] < 4:
            Y.append(0)
        else:
            Y.append(1)

	#prints running time of algorithm
    print("%s seconds" % (time.time() - start_time))

	#prints results
    print ("Results")
    auc = roc_auc_score(totalY, Y)
    print("Area under curve : " + str(auc))
    fpr,tpr, _ = roc_curve(totalY, Y)
    print ("False Positive Rate : " + str(fpr[1]))
    _, recall, _ = precision_recall_curve(totalY, Y)
    print ("Recall : " + str(recall[1]))

	#to plot ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.2f)' % auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()



