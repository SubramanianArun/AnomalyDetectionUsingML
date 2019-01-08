import csv
import numpy
import os,sys
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

csv_file_sample='D:\\class_work\\560\project\\Training Data.csv'
csv_labelled_training='D:\\class_work\\560\project\\Training_labelled.csv'
csv_test_data='D:\\class_work\\560\\project\Testing Data.csv'
kmeans_result='D:\\class_work\\560\\project\Kmeans_test_data.csv'
"""

csv_file_sample='/Users/Sam/Downloads/Training Data.csv'
csv_labelled_training='/Users/Sam/Downloads/Training_labelled.csv'
csv_test_data='/Users/Sam/Downloads/Testing Data.csv'
kmeans_result='/Users/Sam/Downloads/kmeans_result.csv'





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
            if(i==0):
                print(rows)

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
#predict_classes=kmeans.labels_
predict_classes=kmeans.predict(X)
predict_test_classes=kmeans.predict(Y)
#print (predict_classes)



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, predict_classes)
clf.predict(Y)




#print(clf)

#tree.plot(cf)


plt.scatter(Y[:,0], Y[:,1], c=predict_test_classes)#, s=50, cmap='viridis')


centers = kmeans.cluster_centers_
#print (centers)
plt.scatter(centers[:, 0], centers[:, 1], c='black')#, s=200, alpha=0.5);
plt.show()
######
count=0
with open (csv_labelled_training,'w') as wf:
    with open(csv_file_sample,'r') as f:
        csv_writer=csv.writer(wf)
        csv_reader= csv.reader(f,delimiter=',')
        for i,row in enumerate(csv_reader):
            csv_writer.writerow(row+[predict_classes[count]])
            count=count+1

count=0
with open (kmeans_result,'w') as wf:
    with open(csv_test_data,'r') as f:
        csv_writer=csv.writer(wf)
        csv_reader= csv.reader(f,delimiter=',')
        for i,row1 in enumerate(csv_reader):

            csv_writer.writerow(row1+[predict_test_classes[count]])
            count=count+1



print ("all done")
