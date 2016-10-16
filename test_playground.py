import synthea_parse_lib as spb

patient_files = spb.get_patients("../Patient_Records/output/fhir")

key_set = spb.get_biometrics('heartdisease.json','heart disease').keys()

count = 0
for patient in patient_files:
	if spb.get_conditions(patient,'heart disease'):
		key_set_new = []
		count = count + 1
		print "Patient #%d: " %(count)
		vals = spb.get_biometrics(patient,'heart disease')
		val_keys = vals.keys()
		for key1 in val_keys:
			for key2 in key_set:
				if key1.lower() == key2.lower():
					key_set_new.append(key1)
		key_set_new = key_set

for key in key_set:
	print key

