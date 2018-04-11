import csv
import numpy
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

#one time csv file generation from txt file

"""txt_file = ('D:\\class_work\\560\\project\\data1.txt')
csv_file = ('D:\\class_work\\560\\project\\Input_data1.csv')"""
txt_file = '/Users/Sam/Downloads/data1.txt'
csv_file = '/Users/Sam/Downloads/Input_data1.csv'
in_txt = csv.reader(open(txt_file, 'rt'),delimiter=' ')
out_csv = csv.writer(open(csv_file, 'wt'))
out_csv.writerows(in_txt)
#csv_file = 'D:\\class_work\\560\\project\\Input_data.csv'
#csv_file_sample='D:\\class_work\\560\project\\Input_sampled.csv'
#csv_labelled_training='D:\\class_work\\560\project\\Training_labelled.csv'
csv_file_sample='/Users/Sam/Downloads/Input_sampled1.csv'
csv_labelled_training='/Users/Sam/Downloads/Training_labelled1.csv'

data = []
X=numpy.empty([7000,2])
count=0
f1=[]
f2=[]
with open (csv_file_sample,'wt') as wf:
    with open(csv_file,'rt') as f:
        csv_reader= csv.reader(f,delimiter=',')
        for i,row in enumerate(csv_reader):
            if(i<7000):
               # print (i, row)
                data1=(row[4])
                data2=(row[5])
                wf.write(data1)
                wf.write(',')
                wf.write(data2)
                wf.write('\n')
                X[count]=[float(data1),float(data2)]
                f1.append(float(data1))
                f2.append(float(data2))
                #print (X[count])
                count=count+1



 #kmeans for sampled training data


f.close()
wf.close()

#print (f1)

kmeans=KMeans(n_clusters=2).fit(X)
predict_classes=kmeans.predict(X)
#print (kmeans.cluster_centers_)
#print (predict_classes.shape)




plt.scatter(f1[1:10], f2[1:10]  , c='red')#, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black')#, s=200, alpha=0.5);

count=0
with open (csv_labelled_training,'w') as wf:
    with open(csv_file_sample,'r') as f:
        csv_writer=csv.writer(wf)
        csv_reader= csv.reader(f,delimiter=',')
        for i,row in enumerate(csv_reader):
            csv_writer.writerow(row+[predict_classes[count]])
            count=count+1
print ("all done")
