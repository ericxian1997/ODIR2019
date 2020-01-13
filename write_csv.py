import csv

f = open('1.csv','w',encoding='utf-8')

csv_writer = csv.writer(f)

# 3. 构建列表头
csv_writer.writerow(['ID','N','D','G','C','A','H','M','O'])
