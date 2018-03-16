import csv

txt_file = r"D:\class_work\560\project\data.txt"
csv_file = r"D:\class_work\560\project\Input_data.csv"
in_txt = csv.reader(open(txt_file, "rb"), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wb'))
out_csv.writerows(in_txt)
