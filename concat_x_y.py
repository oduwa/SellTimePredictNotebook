import csv, argparse

x = []
y = []
with open('../data/X_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x.append(row[1])

with open('../data/y_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        y.append(row[1])

assert len(x) == len(y)

with open('../data/x_y_test.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(['product_id', 'label'])
    for i in range(len(x)):
        writer.writerow([x[i], y[i]])
