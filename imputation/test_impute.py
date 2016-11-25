from fancyimpute import NuclearNormMinimization, IterativeSVD, MICE
import csv
import numpy as np

# Categorical Columns (filled):
columns = ["Heart Disease", "Male"]


# Initialize:
csv_file = "../synthea_parsed_data.csv"
test_file = "../induct_MICE.csv"
solver = MICE()

# Extract incomplete data:
X_incomplete = []
with open(csv_file,'rb') as csvfile:
	my_reader = csv.reader(csvfile, delimiter=',')
	header = next(my_reader)
	
	male = header.index("Male")
	heart_disease = header.index("Heart Disease")

	for row in my_reader:
		X_incomplete.append(row)

# Separate into Different Imputation Arrays based on categorical Data:
X11 = []
X10 = []
X01 = []
X00 = []

for row in X_incomplete:
	if row[male] == "1" and row[heart_disease] == "1":
		X11.append(row)
	elif row[male] == "1" and row[heart_disease] == "0":
		X10.append(row)
	elif row[male] == "0" and row[heart_disease] == "1":
		X01.append(row)
	elif row[male] == "0" and row[heart_disease] == "0":
		X00.append(row)


my_func = lambda x: float("nan") if x == "" else float(x)


X11 = np.array([map(my_func, sub) for sub in X11])
X10 = np.array([map(my_func, sub) for sub in X10])
X01 = np.array([map(my_func, sub) for sub in X01])
X00 = np.array([map(my_func, sub) for sub in X00])


# Try an algorithm!
X11_imp = solver.complete(X11)
X10_imp = solver.complete(X10)
X01_imp = solver.complete(X01)
X00_imp = solver.complete(X00)

with open(test_file,'wb') as testfile:
	my_writer = csv.writer(testfile,delimiter=',')
	fieldnames = header;

	my_writer.writerow(header)
	for i in range(0,X11_imp.shape[0]):
		my_writer.writerow(X11_imp[i])
	for i in range(0,X10_imp.shape[0]):
		my_writer.writerow(X10_imp[i])
	for i in range(0,X01_imp.shape[0]):
		my_writer.writerow(X01_imp[i])
	for i in range(0,X00_imp.shape[0]):
		my_writer.writerow(X00_imp[i])

