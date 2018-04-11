import csv
import numpy
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

csv_file_sample='/Users/Sam/Downloads/Input_sampled1.csv'
csv_labelled_training='/Users/Sam/Downloads/Training_labelled1.csv'

data = []
X=numpy.empty([7000,2])
count=0
f1=[]
f2=[]
