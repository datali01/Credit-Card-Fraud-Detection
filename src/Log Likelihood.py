import csv
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def LL():
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
    clf = FactorAnalysis()
    clf.fit(totalX)
    #logLik = clf.score(totalX)
    Y = []
    llScores = clf.score_samples(totalX)
    for i in range(len(totalY)):
        if llScores[i] > -60 and llScores[i] < -25:
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
    print ("Running CV on Log Likelihood approach.")
    LL()

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

    #newTotalX = np.fft.fft(totalX)
    totalX = scalar.fit_transform(totalX)
    print ("Data Loaded")
    clf = FactorAnalysis()
    clf.fit(totalX)
    #logLik = clf.score(totalX)
    Y = []
    llScores = clf.score_samples(totalX)						#calculates log likelihood of each sample (instead of average of whole data set)
    for i in range(len(totalY)):
        if llScores[i] > -60 and llScores[i] < -25:
            Y.append(0)
        else:
            Y.append(1)
	#prints running time of algorithm
    print("%s seconds" % (time.time() - start_time))
	#print results
    print ("Results")
    auc = roc_auc_score(totalY, Y)
    print("Area under curve : " + str(auc))
    fpr, tpr, _ = roc_curve(totalY, Y)
    print ("False Positive Rate : " + str(fpr[1]))
    _, recall, _ = precision_recall_curve(totalY, Y)
    print ("Recall : " + str(recall[1]))

	#to plot ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()
