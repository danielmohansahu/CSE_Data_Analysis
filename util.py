# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:48:45 2016

@author: Jiachen
"""

import random
import numpy as np
from sklearn.preprocessing import Imputer


def process(sourcefile, out_test, out_train):
    """
    Reads in sourcefile with blanks for missing entries \n
    Converts blanks into -1
    Randomize row ordering \n
    Using complete rows, extract 1/10 to create test set \n
    Using remaining rows, run imputation to fill -1 with mean values of column
    """
    with open(sourcefile, 'r') as f:
        list_lines = f.readlines()

    header = list_lines[0]

    replace_blank(list_lines)

    randomize_rows(list_lines)
    
    # Note that list_remaining does not contain the header row
    list_remaining = create_test_set(list_lines, out_test)
    
    data = convert_to_numpy(list_remaining)
    
    imputed = impute(data)

    list_train = np.ndarray.tolist(imputed)
    for idx, row in enumerate(list_train):
        s = ','.join(map(str,row)) + '\n'
        list_train[idx] = s
    list_train = [header] + list_train
    
    randomize_rows(list_train)
    with open(out_train, 'w') as f:
        f.writelines(list_train)


def replace_blank(input_list):
    """
    Converts blanks into -1 \n    
    Argument: \n
    input_list - list of strings, comma-separated
    """     
    for idx in range(0, len(input_list)):
        list_entries = input_list[idx].strip().split(',')
        for idx2, entry in enumerate(list_entries):
            if entry == '':
                list_entries[idx2] = '-1'
        s = ','.join(list_entries) + '\n'
        input_list[idx] = s


def randomize_rows(input_list):
    """
    In-place randomization of rows \n
    Argument: \n
    input_list - list of strings
    """
    num_rows = len(input_list)
    for idx in range(1, num_rows-1):
        temp = input_list[idx]
        rand_idx = random.randint(idx+1, num_rows-1)
        input_list[idx] = input_list[rand_idx]
        input_list[rand_idx] = temp
        
    
def create_test_set(input_list, test_file):
    """
    Extracts complete rows from input_list, writes 1/10 into test set. \n
    Remaining 9/10 are recombined with the incomplete rows (those with -1)
    and returned \n

    Argument: \n
    input_list - row-randomized list of strings \n
    test_file - name of file to write the test data \n
    
    Output: \n
    output_list - concatenation of incomplete rows with 9/10 of the complete rows
    """
    header = input_list[0]
    list_incomplete = []
    list_complete = []
    for line in input_list[1:]:
        list_entries = line.strip().split(',')
        if '-1' in list_entries:
            list_incomplete.append(line)
        else:
            list_complete.append(line)
            
    num_complete = len(list_complete)
    f = open(test_file, 'w')
    f.write(header)
    f.writelines( list_complete[int(num_complete/10.0*9):] )
    f.close()    

    return list_incomplete + list_complete[0:int(num_complete/10.0*9)]


def convert_to_numpy(list_strings):

    num_row = len(list_strings)
    num_col = list_strings[0].count(',') + 1
    data = np.zeros((num_row, num_col), dtype=np.float32)
    
    idx = 0    
    for line in list_strings:
        list_token = map(float, line.strip().split(','))
        data[idx] = list_token
        idx += 1
    
    return data
        

def impute(data):
    """
    Reads sourcefile, imputes all -1 entries using average of column, writes \n
    to outfile.
    """
    # Compute some values manually to check correctness
    # First column
    col = data[:,0]
    # Index of first occurrence of -1
    itemindex = np.where(col == -1)
    idx_first = itemindex[0][0]
    # Remove all -1
    col = col[col != -1]
    # Calculate mean
    avg = np.mean(col)
    
    im = Imputer(missing_values=-1, strategy='mean', axis=0)
    im = im.fit(data)
    data = im.transform(data)

    col = data[:,0]
    if avg == col[idx_first]:
        print "Passed check"
    else:
        print "Failed check"        
        
    return data
    

# The functions below are meant to be used individually #

def randomize_data(sourcefile, outfile):
    """
    Read in data, randomize rows and write to file
    """
    with open(sourcefile, 'r') as f:
        list_lines = f.readlines()
    
    num_rows = len(list_lines)
    # In-place randomization
    for idx in range(1, num_rows-1):
        temp = list_lines[idx]
        # Random index from current index to end of list
        rand_idx = random.randint(idx+1, num_rows-1)
        # Swap
        list_lines[idx] = list_lines[rand_idx]
        list_lines[rand_idx] = temp
    
    with open(outfile, 'w') as f:
        f.writelines(list_lines)


def remove_blanks(sourcefile, outfile):
    """
    Reads sourcefile, writes all rows that do not contain blanks into outfile
    """
    f = open(sourcefile,'r')
    list_lines = f.readlines()
    f.close()
    
    f = open(outfile,'w')
    for line in list_lines:
        list_entries = line.strip().split(',')
        # If there are no empty entries in row
        if '' not in list_entries:
            f.write(line)
    f.close()
    
    
def fill_blank(sourcefile, outfile):
    """
    Reads sourcefile, fills all blanks with -1, writes to outfile
    """
    f = open(sourcefile, 'r')
    list_lines = f.readlines()
    f.close()
    
    f = open(outfile, 'w')
    for line in list_lines:
        list_entries = line.strip().split(',')
        for idx, entry in enumerate(list_entries):
            if entry == '':
                list_entries[idx] = '-1'
        s = ','.join(list_entries) + '\n'
        f.write(s)
    f.close() 
    
    
def gen_train_test(sourcefile, train_file, test_file):
    """
    Reads sourcefile, writes the first 90% into train_file,
    writes the last 10% into test_file
    """
    f = open(sourcefile, 'r')
    list_lines = f.readlines()
    f.close()
    
    num_lines = len(list_lines)
    f = open(train_file, 'w')
    f.writelines( list_lines[0: int(num_lines/10.0*9)] )
    f.close()
    
    f = open(test_file, 'w')
    f.write( list_lines[0] ) # need header line
    f.writelines( list_lines[int(num_lines/10.0*9):] )
    f.close()