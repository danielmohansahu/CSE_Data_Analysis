import csv
import synthea_parse_lib as spb
import os

# Load Biometric Data from Patient Files and Load into CSV
files_loc = "../../Patient_Data/Heart_Disease"

patient_files = spb.get_patients(files_loc)

count = 0
for patient in patient_files:
	count = count + 1

	# Check if the patient has heart disease from .csv file


	conds = spb.get_biometrics_gen(patient,'heart disease')

	print(patient)
	print(conds)

	raw_input("wait : ")

