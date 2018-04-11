import csv
import numpy
import sys
import random


#print (os.getcwd())
"""txt_file = '/Users/Sam/Downloads/data.txt'
csv_file = '/Users/Sam/Downloads/Input_data.csv'
in_txt = csv.reader(open(txt_file, 'rb'), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wb'))
out_csv.writerows(in_txt)"""
csv_file = '/Users/Sam/Downloads/Input_data.csv'
csv_file_sample='/Users/Sam/Downloads/Input_sampled.csv'
randomnumber = []
for count in range(0,10014):
    rand = random.randint(0,2300000)
    randomnumber.append(rand)
print len(randomnumber)
with open (csv_file_sample,'wf') as wf:
    with open(csv_file,'rb') as f:
        csv_reader= csv.reader(f,delimiter=',')
        for i,row in enumerate(csv_reader):
            if i in randomnumber:
                print i
                data1=(row[4])
                data2=(row[5])
                wf.write(data1)
                wf.write(',')
                wf.write(data2)
                wf.write('\n')
    f.close()
wf.close()
