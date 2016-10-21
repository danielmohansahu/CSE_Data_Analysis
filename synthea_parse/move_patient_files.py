import synthea_parse_lib as spb
from shutil import copy

patient_files = spb.get_patients("../../Patient_Records/output/fhir")

count = 0
hd = 0
nhd = 0
for patient in patient_files:
	count = count + 1
	if spb.has_condition(patient,'heart disease'):
		hd = hd + 1
		copy(patient,"../Patient_Data/Heart_Disease/")
	elif nhd < 1000:
		try:
			len(spb.get_biometrics_gen(patient).keys())		# To see if the function worked
			nhd = nhd + 1
			copy(patient,"../Patient_Data/Healthy/")
		except:
			print "Skipping a healthy patient! Bias? ..."
	print "Patient #%d/%d " %(count,len(patient_files))


print "Total of %d / %d people with heart disease" %(hd,count)
