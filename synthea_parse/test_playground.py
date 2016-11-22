import synthea_parse_lib as spb
from shutil import copy

print("Collecting patient files...\n")
patient_files = spb.get_patients("../../Patient_Records_New/Output/fhir")

count = 0
hd1 = 0
hd2 = 0
hd3 = 0
hd4 = 0
for patient in patient_files:
	count = count + 1
	if spb.get_keyword(patient,"53741008"):
		hd1 = hd1 + 1;
		print "%d patients with 53741008" %(hd1)
	elif spb.get_keyword(patient,"72092001"):
		hd2 = hd2 + 1;
		print "%d patients with 72092001" %(hd2)
	elif spb.get_keyword(patient,"414024009"):
		hd3 = hd3 + 1;
		print "%d patients with 414024009" %(hd3)
	elif spb.get_keyword(patient,"128599005"):
		hd4 = hd4 + 1;
		print "%d patients with 128599005" %(hd4)
	else:
		pass

print "----------------------------------------------"
print "%d patients with 53741008" %(hd1)
print "%d patients with 72092001" %(hd2)
print "%d patients with 414024009" %(hd3)
print "%d patients with 128599005" %(hd4)
print "%d patients with heart disease." %(hd1 + hd2 + hd3 + hd4)