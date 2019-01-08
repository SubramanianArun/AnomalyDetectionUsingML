

#Libraries for analysis
import csv
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import svm
from sklearn import tree
from sklearn.cluster import DBSCAN
from sklearn import metrics

#Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

#File Paths
training_data_file = '/Users/Sam/Downloads/Training Data.csv'
training_output = '/Users/Sam/Downloads/training_svm_label.csv'
testing_data_file = '/Users/Sam/Downloads/Testing Data.csv'
testing_output = '/Users/Sam/Downloads/testing_svm_label.csv'

accuracy_output = '/Users/Sam/Downloads/accuracy_svm_label.csv'

#Variables

#Import data
training = pd.read_csv(training_data_file)
X = training.as_matrix()
testing = pd.read_csv(testing_data_file)
Y = testing.as_matrix()





#SVM Clustering - OneClassSVM
clf = svm.OneClassSVM(nu = 0.8, kernel = 'linear')
clf.fit(X)
x_pred =clf.predict(X)


"""
#SVM Training - SVC
clf = svm.SVC(kernel = 'linear', C = 1.0)
clf.fit(X,x_pred)
x_class = clf.predict(X)


#Plot the training data
#sns.lmplot('Temp', 'Humidity', data=training, hue='Temp', palette = 'Set1', fit_reg=False, scatter_kws={"s":70})
#plt.show()
#pl.set_cmap(pl.cm.Paired)
pl.scatter(X[x_pred>0,0], X[x_pred>0,1], c='blue', label='cluster 1')
pl.scatter(X[x_pred<=0,0], X[x_pred<=0,1], c='red', label='cluster 2')
pl.title('SVM - Training data')
pl.axis('tight')
plt.legend()
pl.show()

"""

#Testing Data
y_pred = clf.predict(Y)

print y_pred[:20]
db_classes = clf.predict(Y)

#Normalizing the output of SVM for comparison
for i in range(0, len(y_pred)):
    if y_pred[i]<= 0:
        y_pred[i] = 0
    else:
        y_pred[i] = 1

#Ground Truth prep

for i in range(0,len(db_classes)):
    if Y[i,0] > 37:
        db_classes[i] = 1
    else:
        db_classes[i] = 0

print db_classes[:20]

accuracy_svm=metrics.accuracy_score(db_classes,y_pred)
print accuracy_svm
m1=metrics.silhouette_score(Y,y_pred,metric='euclidean')
print m1
m2 = metrics.confusion_matrix(db_classes,y_pred,sample_weight=None,labels=None)
print m2
#DBSCAN
#db_classes=DBSCAN().fit_predict(Y,y=None,sample_weight=None)


count = 0
with open (accuracy_output,'w') as wf:
    csv_writer=csv.writer(wf, delimiter=',')
    for i in range(0,len(y_pred)):
        csv_writer.writerow([y_pred[count]])
        count += 1

#Writing testing label to a CSV file
count=0
try:
    with open (testing_output,'w') as wf:
        with open(testing_data_file,'r') as f:
            csv_writer=csv.writer(wf)
            csv_reader= csv.reader(f,delimiter=',')
            for i,row in enumerate(csv_reader):
                csv_writer.writerow(row+[y_pred[count]])
                count = count+1
except IndexError:
    pass

#Plot the testing data
#sns.lmplot('Temp', 'Humidity', data=training, hue='Temp', palette = 'Set1', fit_reg=False, scatter_kws={"s":70})
#plt.show()
#pl.set_cmap(pl.cm.Paired)
"""
pl.scatter(Y[y_pred>0,0], Y[y_pred>0,1], c='white', label='cluster 1')
pl.scatter(Y[y_pred<=0,0], Y[y_pred<=0,1], c='black', label='cluster 2')
pl.axis('tight')
pl.title('SVM - Testing data')
plt.legend()
pl.show()
"""

#Writing training label to a CSV file
count=0
try:
    with open (training_output,'w') as wf:
        with open(training_data_file,'r') as f:
            csv_writer=csv.writer(wf)
            csv_reader= csv.reader(f,delimiter=',')
            for i,row in enumerate(csv_reader):
                csv_writer.writerow(row+[x_pred[count]])
                count=count+1
except IndexError:
    pass
