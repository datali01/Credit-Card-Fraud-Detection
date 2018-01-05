import csv
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.stats import chisquare
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#For Cross Validation
def StrChi():
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
    conf = 0.99
    for k in range(10):
        Y = []
        for i in range(len(totalX)):
            _, p = chisquare(totalX[i])							#p-value returned from ChiSquare Test
            if p < (1.0 - conf):
                Y.append(1)
            else:
                Y.append(0)
        print (conf)
        auc = roc_auc_score(totalY, Y)
        print(auc)
        fpr, _, _ = roc_curve(totalY, Y)
        print (fpr[1])
        _, recall, _ = precision_recall_curve(totalY, Y)
        print (recall[1])

        conf -= 0.01
    return auc, fpr[1], recall
	
#Main Function to do Cross validation followed by Testing
def main():
    totalX = []
    totalY = []
    flag = True
    countTrain = 0
    print ("Running CV on StrOUD with ChiSquare Test.")
    StrChi()
    with open("creditcard.csv", "rb") as f:		#randomly shuffled and standardized data set
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
    conf = 0.915												#best confidence value from CV

    Y = []
    for i in range(len(totalX)):
        _, p = chisquare(totalX[i])
        if p < (1.0 - conf):
            Y.append(1)
        else:
            Y.append(0)
    auc = roc_auc_score(totalY, Y)
    print(auc)
    fpr, _, _ = roc_curve(totalY, Y)
    print (fpr[1])
    _, recall, _ = precision_recall_curve(totalY, Y)
    print (recall[1])

if __name__ == '__main__':
    main()