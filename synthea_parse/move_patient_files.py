import synthea_parse_lib as spb
from shutil import copy
import csv
import os

print("Collecting patient files...\n")
patient_files = spb.get_patients("../../Patient_Records_Final/output/fhir")

count = 0
hd = 0
nhd = 0
for patient in patient_files:
	count = count + 1

	# Check if the patient has heart disease from .csv file
	filename, file_extension = os.path.splitext(patient)
	HD = False;
	with open(filename + ".csv") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			HD = HD or row['Heart_Disease'] == 'true'

	if HD:
		hd = hd + 1
		copy(patient,"../../Patient_Data/Heart_Disease/")
		copy(filename + ".csv","../../Patient_Data/Heart_Disease/")
	elif nhd < hd:
		try:
			# len(spb.get_biometrics_gen(patient).keys())		# To see if the function worked. Bad idea?
			copy(patient,"../../Patient_Data/Healthy/")
			copy(filename + ".csv","../../Patient_Data/Healthy/")
			nhd = nhd + 1
		except:
			print "Skipping a healthy patient! Bias? ..."
	print "Patient #%d/%d; %d with heart disease" %(count,len(patient_files),hd)
