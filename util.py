# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:48:45 2016

@author: Jiachen
"""

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