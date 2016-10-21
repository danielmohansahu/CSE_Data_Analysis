import json
import os
import fnmatch
from datetime import datetime

# # # # # # # # # # ALL KEYS IN DATA # # # # # # # # # # 
def get_all_keys(filename):
  with open(filename) as file :
    # Check file extension:
    try:
      data = json.load(file)
      data = data['entry']
    except TypeError as e:
      print e
  
    return_vals = [];

    # Iterate through data
    for i in range(0,len(data)):
      return_vals.append(data[i]['resource'].keys())
  
    return set([item for sublist in return_vals for item in sublist])

# # # # # # # # # # PATIENT BIOMETRICS + METADATA # # # # # # # # # # 
# Return biometrics for patient with heart disease (specifically)
def get_biometrics(filename,condition):
  biometric_keys = [
    'multipleBirthBoolean',
    'maritalStatus',
    'birthDate',
    'gender',
    'extension']
  return_vals = {}
  condition_dict = {
    "heart disease":"53741008",
    "diabetes":"44054006",
    "prediabetes":"15777000",
    "hypertension":"38341003"}

  with open(filename) as file :
    # Check file extension:
    try:
      data = json.load(file)
      data = data['entry']
      data_general = data[0]['resource']  # Location of biometric data
    except TypeError as e:
      print e
  
  # General Patient Data:
  try:
    return_vals['multipleBirthBoolean'] = data_general['multipleBirthBoolean']
  except: pass
  try:
    return_vals['maritalStatus'] = data_general['maritalStatus']['coding'][0]['code']
  except: pass
  try:
    return_vals['gender'] = data_general['gender']
  except: pass
  try:
    return_vals['race'] = data_general['extension'][0]['valueCodeableConcept']['coding'][0]['display']
  except: pass
  try:
    return_vals['ethnicity'] = data_general['extension'][1]['valueCodeableConcept']['coding'][0]['display']
  except: pass

  # Append test results for encounter with first diagnosis of heart disease
  # ASSUMING ONLY FIRST ENCOUNTER HAS A DIAGNOSIS
  encounter_id = 0
  for i in range(0,len(data)):
    if data[i]['resource']['resourceType'].lower() == 'condition':
      for j in range(0,len(data[i]['resource']['code']['coding'])):
        if data[i]['resource']['code']['coding'][j]['code'].lower() == condition_dict[condition]:
          encounter_id = (data[i]['resource']['context']['reference'])
          return_vals['age'] = datetime.strptime(data[i]['resource']['onsetDateTime'][0:10],'%Y-%m-%d') - datetime.strptime(data_general['birthDate'],'%Y-%m-%d')
  
  if encounter_id == 0:
    print "No diagnosis of this condition!"
  else:
    for i in range(0,len(data)):
      if data[i]['resource']['resourceType'].lower() == 'observation':
        if data[i]['resource']['encounter']['reference'] == encounter_id:
          try:
            return_vals[data[i]['resource']['code']['coding'][0]['display']] = str(data[i]['resource']['valueQuantity']['value']) + " " + data[i]['resource']['valueQuantity']['unit']
          except:
            try:
              for k in range(0,len(data[i]['resource']['component'])):
                return_vals[data[i]['resource']['component'][k]['code']['coding'][0]['display']] = str(data[i]['resource']['component'][k]['valueQuantity']['value']) + " " + data[i]['resource']['component'][k]['valueQuantity']['unit']
            except:
              pass   
  return return_vals

# # # # # # # # # # PATIENT CONDITIONS # # # # # # # # # #
# Returns keywords i.e. "diabetic","opiod addict", "hear disease", etc. 
def get_conditions(filename):
  with open(filename) as file :
    # Check file extension:
    try:
      data = json.load(file)
      data = data['entry']
    except TypeError as e:
      print e
  
    return_vals = [];

    # Iterate through data
    if recursive_strfind(data,'44054006'):
      return_vals.append('heart disease')
    if recursive_strfind(data,'53741008'):
      return_vals.append('diabetes')
    if recursive_strfind(data,'15777000'):
      return_vals.append('prediabetes')

    return return_vals

# Returns true if the given patient has the condition specified in keyword 
def has_condition(filename,keyword):
  with open(filename) as file :
    # Check file extension:
    try:
      data = json.load(file)
      data = data['entry']
    except TypeError as e:
      print e
  
    # Supported keywords:
    condition_dict = {
      "heart disease":"53741008",
      "diabetes":"44054006",
      "prediabetes":"15777000",
      "hypertension":"38341003"}

    # Iterate through data
    if recursive_strfind(data,condition_dict[keyword]):
      return True
    else:
      return False

# # # # # # # # # # KEYWORDS # # # # # # # # # #
# Merely searches the file for a keyword
def get_keyword(filename,string):
  with open(filename) as file :
    # Check file extension:
    try:
      data = json.load(file)
      data = data['entry']
    except TypeError as e:
      print e

  return recursive_strfind(data,string) > 0


# # # # # # # # # # PATIENT CONDITIONS # # # # # # # # # #
# Searches every subentry of the given datum for the specified string:
def recursive_strfind(datum,string):
  return recursive_strfind_inner(0,datum,string) > 0

# Blegh:
def recursive_strfind_inner(counter,datum,string):
  # Break condition:
  if counter:
    return counter
  else:
    # Check if the datum is numeric:
    try:
      datum*1.0
    except:
      # Check if the datum is a string and has our keyword
      if isinstance(datum,basestring):
        if datum.lower().find(string) >= 0:
          counter = counter + 1
      # Check if it's a list/set/tuple
      else:
        # Treat it as a list:
        try:
          for i in range(0,len(datum)):
            counter = recursive_strfind_inner(counter,datum[i],string)
        # Treat it as a set:
        except:
          for val in datum:
            counter = recursive_strfind_inner(counter,datum[val],string)

  return counter

# # # # # # # # # # PATIENT CONDITIONS # # # # # # # # # #
# Obtain the fullpath of every .json file in the given directory:
# Caution: pretty slow sometimes
def get_patients(directory):
  return_vals = []
  for root, dirnames, filenames in os.walk(directory):
    for filename in fnmatch.filter(filenames, '*.json'):
      return_vals.append(os.path.join(root, filename))
  return return_vals

