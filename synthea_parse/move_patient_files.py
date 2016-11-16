import synthea_parse_lib as spb
from shutil import copy

print("Collecting patient files...\n")
patient_files = spb.get_patients("../../Patient_Records_New/Output/fhir")

count = 0
hd = 0
nhd = 0
for patient in patient_files:
	count = count + 1
	if spb.has_condition(patient,'heart disease'):
		hd = hd + 1
		copy(patient,"../Patient_Data/Heart_Disease/")
	elif nhd < 2116:
		try:
			len(spb.get_biometrics_gen(patient).keys())		# To see if the function worked
			copy(patient,"../Patient_Data/Healthy/")
			nhd = nhd + 1
		except:
			print "Skipping a healthy patient! Bias? ..."
	print "Patient #%d/%d; %d with heart disease" %(count,len(patient_files),hd)

