import csv
import synthea_parse_lib as spb
import os

# Load Biometric Data from Patient Files and Load into CSV
# files_loc = "../../Patient_Data/"
# csv_loc = "../synthea_parsed_data.csv"

files_loc = "../../synthea/output/"
csv_loc = "../testing/synthea_parsed_data.csv"

print "Gathering patient files..."
patient_files = spb.get_patients(files_loc)

# Create Header:
header_idx = {'Systolic Blood Pressure': 'Systolic BP (mmHg)', 
'Diastolic Blood Pressure': 'Diastolic BP (mmHg)',
'Total Cholesterol': 'Total Cholesterol (mg/dl)',
'Low Density Lipoprotein Cholesterol': 'LDL Cholesterol (mg/dl)',
'High Density Lipoprotein Cholesterol': 'HDL Cholesterol (mg/dl)',
'Triglycerides': 'Triglycerides (mg/dl)',
'Sodium':'Sodium (mmol/L)',
'Body Weight': 'Body Weight (kg)',
'Chloride': 'Chloride (mmol/L)',
'Glucose': 'Glucose (mg/dl)',
'Potassium': 'Potassium (mmol/L)',
'Urea Nitrogen': 'Urea Nitrogen (mg/dl)',
'Hemoglobin A1c/Hemoglobin.total in Blood': 'Hemoglobin A1c (%)',
'Body Mass Index': 'Body Mass Index (kg/m2)',
'Carbon Dioxide': 'Carbon Dioxide ()',
'Calcium': 'Calcium (mg/dl)',
'Female': 'Female',
'Male': 'Male',
'age': 'Age (years)',
'Body Height': 'Body Height (cm)',
'Creatinine': 'Creatinine (mg/dl)',
'Smoker': 'Smoker',
'Heart Disease': 'Heart Disease'}

headers = {}
count = 0
with open(csv_loc,'wb') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=header_idx.keys(), extrasaction='ignore')

	# Write header (from header_idx specified above)
	for n in writer.fieldnames:
		headers[n] = header_idx[n]
	writer.writerow(headers)

	for patient in patient_files:
		count = count + 1

		# Check if the patient has heart disease from .csv file
		filename, file_extension = os.path.splitext(patient)
		HD = False;
		Smoke = '';
		with open(filename + ".csv") as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				HD = HD or row['Heart_Disease'] == 'true'
				if row['Smoker'] == 'true':
					Smoke = 1
				elif row['Smoker'] == 'false':
					Smoke = 0

		if spb.has_condition(patient,'heart disease'):
			conds = spb.get_biometrics(patient,'heart disease')
		else:
			conds = spb.get_biometrics_gen(patient)

		if HD:
			conds["Heart Disease"] = 1
		else:
			conds["Heart Disease"] = 0
		conds["Smoker"] = Smoke

		if conds["gender"] == "male":
			conds["Male"] = 1
			conds["Female"] = 0
		elif conds["gender"] == "female":
			conds["Male"] = 0
			conds["Female"] = 1

		print "Writing patient %d" %(count)
		writer.writerow(conds)
