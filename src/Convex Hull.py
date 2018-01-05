import csv
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def Convex():
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
            countTrain += 1
            if countTrain > 228000:								#doesn't use the last 20% of data
                break
            totalX.append([float(i) for i in row[:-1]])
            totalY.append(int(row[-1]))
    totalX = scalar.fit_transform(totalX)
    print ("CV Data Loaded.")
    pca = PCA(n_components=11)
    newX = pca.fit_transform(totalX)
    hull = ConvexHull(newX)
    # print (hull.vertices)
    Y = [0] * len(totalX)
    for i in hull.vertices:
        Y[i] = 1

    auc = roc_auc_score(totalY, Y)
    print(auc)
    fpr, _, _ = roc_curve(totalY, Y)
    print (fpr[1])
    _, recall, _ = precision_recall_curve(totalY, Y)
    print (recall[1])

#Main Function to do Cross validation followed by Testing
def main():
    totalX = []
    totalY = []
    flag = True
    countTrain = 0
    print ("Running CV on Convex Hull.")
    Convex()

    start_time = time.time()
    with open("creditcard.csv", "rb") as f:
        data = csv.reader(f)
        for row in data:
            if flag:
                flag = False
                continue
            countTrain += 1
            if countTrain > 228000:
                totalX.append([float(i) for i in row[:-1]])
                totalY.append(int(row[-1]))
    totalX = scalar.fit_transform(totalX)
    print ("Test Data Loaded.")
    pca = PCA(n_components=11)
    newX = pca.fit_transform(totalX)
    hull = ConvexHull(newX)
    Y = [0] * len(totalY)

    for i in hull.vertices:
        Y[i] = 1
	
	#prints running time of algorithm
    print("%s seconds" % (time.time() - start_time))
	#prints results
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