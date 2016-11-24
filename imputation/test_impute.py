from fancyimpute import NuclearNormMinimization, IterativeSVD, MICE
import csv
import numpy as np

csv_file = "../synthea_parsed_data.csv"

# Solver choice:
#solver = NuclearNormMinimization(min_value = 1.0, error_tolerance=0.0005)
#solver = IterativeSVD()

test_file = "../induct_MICE.csv"
solver = MICE()



X_incomplete = []
with open(csv_file,'rb') as csvfile:
	my_reader = csv.reader(csvfile, delimiter=',')
	header = next(my_reader)

	for row in my_reader:
		X_incomplete.append(row)

X_incomplete = np.array(X_incomplete)

my_func = lambda x: float("nan") if x == "" else float(x)

X_induct = np.zeros(X_incomplete.shape)
for i in range(0,X_incomplete.shape[0]):
	for j in range(0,X_incomplete.shape[1]):
		X_induct[i,j] = my_func(X_incomplete[i,j])


# Try an algorithm!
X_test = solver.complete(X_induct)

with open(test_file,'wb') as testfile:
	my_writer = csv.writer(testfile,delimiter=',')
	fieldnames = header;

	my_writer.writerow(header)
	for i in range(0,X_test.shape[0]):
		my_writer.writerow(X_test[i])

