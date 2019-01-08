

#Libraries for analysis
import csv
import pandas as pd
import numpy as np
import pylab as pl
import graphviz
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
testing_output = '/Users/Sam/Downloads/testing_svm_entropy_label.csv'
decision_tree_dot = '/Users/Sam/Downloads/dtree_svm_entropy.dot'

accuracy_output = '/Users/Sam/Downloads/accuracy_svmwithDT_label.csv'
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
print x_pred


"""
#Plot the training data
#sns.lmplot('Temp', 'Humidity', data=training, hue='Temp', palette = 'Set1', fit_reg=False, scatter_kws={"s":70})
#plt.show()
#pl.set_cmap(pl.cm.Paired)
pl.scatter(X[x_pred>0,0], X[x_pred>0,1], c='blue', label='cluster 1')
pl.scatter(X[x_pred<=0,0], X[x_pred<=0,1], c='red', label='cluster 2')
pl.axis('tight')
pl.legend()
pl.show()
"""
#Decision Tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_features=2)
clf = clf.fit(X,x_pred)
dt_result = clf.predict(Y)

db_classes = clf.predict(Y)

#Normalizing the output of SVM for comparison
for i in range(0, len(dt_result)):
    if dt_result[i]<= 0:
        dt_result[i] = 0
    else:
        dt_result[i] = 1

#Ground Truth prep

for i in range(0,len(db_classes)):
    if Y[i,0] > 37:
        db_classes[i] = 1
    else:
        db_classes[i] = 0

print db_classes[:20]

#DBSCAN

accuracy_svm_dt=metrics.accuracy_score(db_classes,dt_result)
print dt_result[:20]
print accuracy_svm_dt


m1=metrics.silhouette_score(Y,dt_result,metric='euclidean')
print m1
m2 = metrics.confusion_matrix(db_classes,dt_result,sample_weight=None,labels=None)
print m2








"""
count = 0
with open (accuracy_output,'w') as wf:
    csv_writer=csv.writer(wf, delimiter=',')
    for i in range(0,len(dt_result)):
        csv_writer.writerow([dt_result[count]])
        count += 1
"""





with open(decision_tree_dot,"w") as dt_f:
    dot_data = tree.export_graphviz(clf, out_file=dt_f)
    graph = graphviz.Source(dot_data)



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
