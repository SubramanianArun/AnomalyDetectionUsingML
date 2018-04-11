import csv
import numpy
import os,sys
import graphviz
#import pydot
#import pydotplus
from scipy import stats
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import tree


#one time csv file generation from txt file

"""
txt_file = ('D:\\class_work\\560\\project\\data.txt')
csv_file = ('D:\\class_work\\560\\project\\Input_data.csv')
in_txt = csv.reader(open(txt_file, 'rt'), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wt'))
out_csv.writerows(in_txt)
"""


"""
csv_file_sample='D:\\class_work\\560\project\\Training Data.csv'
csv_labelled_training='D:\\class_work\\560\project\\Training_labelled.csv'
csv_test_data='D:\\class_work\\560\\project\Testing Data.csv'
"""

csv_file_sample='/Users/Sam/Downloads/Training Data.csv'
csv_labelled_training='/Users/Sam/Downloads/Training_labelled.csv'
csv_test_data='/Users/Sam/Downloads/Testing Data.csv'
decision_tree_txt='/Users/Sam/Downloads/decision_tree.dot'
csv_labelled_result='/Users/Sam/Downloads/decisiontree_result.csv'



data = []
X=numpy.empty([6700,2])
count=0
f1=[]
f2=[]

with open(csv_file_sample,'rt') as f:
    csv_reader= csv.reader(f,delimiter=',')
    for i,row in enumerate(csv_reader):

        try:

            data1=(float(row[0]))
            data2=(float(row[1]))
            X[count]=[data1,data2]
            f1.append(data1)
            f2.append(data2)
            #print (X[count])
            count=count+1
        except:
            print ("on line",i)
            print(row[0])
            print(row[1])


count1=0
Y=numpy.empty([3300,2])
with open(csv_test_data,'r') as f:
    csv_reader= csv.reader(f,delimiter=',')
    for i,rows in enumerate(csv_reader):

        try:

            data1=(float(rows[0]))
            data2=(float(rows[1]))
            Y[count1]=[data1,data2]
            count1=count1+1
        except:
            print ("on line",i)
            print(" "+rows[0]+" "+rows[1])




 #kmeans for sampled training data


f.close()
#wf.close()

#print (f1)
#print(X)
kmeans=KMeans(n_clusters=2, random_state=0 ).fit(X)
predict_classes=kmeans.predict(X)
#print (predict_classes)




clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, predict_classes)
dt_result=clf.predict([22.3,-3.9])

print(dt_result)


"""
#print(clf)

#tree.plot(cf)

with open(decision_tree_txt,"w") as dt_f:
    dot_data = tree.export_graphviz(clf, out_file=dt_f)
    graph = graphviz.Source(dot_data)
    #graph.write_png('somefile.png')
#graph.render("DT")
#(graph,) = pydot.graph_from_dot_file(decision_tree_txt)


count=0
with open (csv_labelled_result,'w') as wf:
    with open(csv_test_data,'r') as f:

        csv_writer=csv.writer(wf)
        csv_reader= csv.reader(f,delimiter=',')
        for i,row in enumerate(csv_reader):
            csv_writer.writerow(row+[dt_result[count]])
            count=count+1


#plt.scatter(f1, f2, c='red')#, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
#print (centers)
#plt.scatter(centers[:, 0], centers[:, 1], c='black')#, s=200, alpha=0.5);
######
count=0
with open (csv_labelled_training,'w') as wf:
    with open(csv_file_sample,'r') as f:
        csv_writer=csv.writer(wf)
        csv_reader= csv.reader(f,delimiter='\t')
        for i,row in enumerate(csv_reader):
            csv_writer.writerow(row+[predict_classes[count]])
            count=count+1


"""


print ("all done")
