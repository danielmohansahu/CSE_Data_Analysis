import csv
import synthea_parse_lib as spb

# Load Biometric Data from Patient Files and Load into CSV
files_loc = "../../Patient_Data/"
csv_loc = "../synthea_parsed_data.csv"

patient_files = spb.get_patients(files_loc)

# Create Header:
header = ['number','Heart Disease']
for patient in patient_files:
	# Kinda slow :/
	if spb.has_condition(patient,'heart disease'):
		keys = spb.get_biometrics(patient,'heart disease').keys()
		header = list(set(header + keys))

print "Done with header"

count = 0
with open(csv_loc,'wb') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=header)

	writer.writeheader()
	for patient in patient_files:
		count = count + 1

		if spb.has_condition(patient,'heart disease'):
			conds = spb.get_biometrics(patient,'heart disease')
			conds["number"] = count
			conds["Heart Disease"] = 1
			writer.writerow(conds)
		else:
			conds = spb.get_biometrics_gen(patient)
			conds["number"] = count
			conds["Heart Disease"] = 0
			writer.writerow(conds)


