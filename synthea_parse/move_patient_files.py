import synthea_parse_lib as spb
from shutil import copy
from distutils.dir_util import copy_tree
import csv
import os

synthea_output_path = "../../synthea/output/"
new_file_temp_path = "../../Patient_Records_New/output"
final_file_path = "../../Patient_Records_Final/output/"

# Move from synthea output folder to 
copy_tree(synthea_output_path,new_file_temp_path)


"""

print("Collecting patient files...\n")

# Total Collection:
#patient_files = spb.get_patients("../../Patient_Records_Final/output/fhir")
# Only new files (faster):
patient_files = spb.get_patients("../../Patient_Records_New/output/fhir")

count = 0
hd = 0
nhd = 0
for patient in patient_files:
	count = count + 1

	try:
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
	except:
		print("SKIPPING PATIENT FOR UNKNOWN ERROR!")


"""