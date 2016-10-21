import synthea_parse_lib as spb
from shutil import copy

patient = "heartdisease.json"

vals = spb.get_biometrics_gen(patient)
