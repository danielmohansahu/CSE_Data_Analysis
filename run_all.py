# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:19:39 2016

@author: Jiachen
"""

import model

case = 3

if case == 1:
    
    exp = model.model('data/synthea_imputed_train.csv', 'data/synthea_imputed_test.csv')
    
    params = exp.train_gridsearch(1)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nRandom forests\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
    
    params = exp.train_gridsearch(2)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nAdaboost using decision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
            
    params = exp.train_gridsearch(3)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nGradient boosting using decision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
        
elif case == 2:
    exp = model.model('data/synthea_common_vars_v2.csv', 'data/UCI_common_vars_v2.csv')
    
    params = exp.train_gridsearch(0)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nTrain on synthea, test on Cleveland\n')
        f.write('\nDecision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))    
    
    params = exp.train_gridsearch(1)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nRandom forests\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
    
    params = exp.train_gridsearch(2)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nAdaboost using decision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
            
    params = exp.train_gridsearch(3)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nGradient boosting using decision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
            
elif case == 3:
    exp = model.model('data/uci_data_train_v1.csv', 'data/uci_data_test_v1.csv')
    
    params = exp.train_gridsearch(0)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nTrain on cleveland, test on cleveland\n')
        f.write('\nDecision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))    
    
    params = exp.train_gridsearch(1)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nRandom forests\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
    
    params = exp.train_gridsearch(2)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nAdaboost using decision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))
            
    params = exp.train_gridsearch(3)
    with open('cross_val_results.txt', 'a') as f:
        f.write('\nGradient boosting using decision tree\n')
        for key, value in params.iteritems():
            f.write('%s,%d\n' % (key, value))            
    