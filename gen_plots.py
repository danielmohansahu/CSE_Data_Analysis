# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 12:59:33 2016

@author: Jiachen
"""

import model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



def plot_roc():
    """
    fpr - Element i is the false positive rate of predictions with score >= thresholds[i]
    tpr - Element i is the true positive rate of predictions with score >= thresholds[i]
    threshold - Decreasing thresholds on the decision function used to compute fpr and tpr.
    """    
    
    ss = model.model(use_hardcode=1,choice=1)
    ss.train(3)

    sc = model.model(use_hardcode=1,choice=2)
    sc.train(4)

    cc = model.model(use_hardcode=1,choice=3)
    cc.train(5)    
    
    probs_ss = ss.clf.predict_proba(ss.data_test)
    preds_ss = probs_ss[:,1]
    fpr_ss, tpr_ss, threshold_ss = roc_curve(ss.target_test, preds_ss)
    roc_auc_ss = auc(fpr_ss, tpr_ss)

    probs_sc = sc.clf.predict_proba(sc.data_test)
    preds_sc = probs_sc[:,1]
    fpr_sc, tpr_sc, threshold_sc = roc_curve(sc.target_test, preds_sc)
    roc_auc_sc = auc(fpr_sc, tpr_sc)

    probs_cc = cc.clf.predict_proba(cc.data_test)
    preds_cc = probs_cc[:,1]
    fpr_cc, tpr_cc, threshold_cc = roc_curve(cc.target_test, preds_cc)
    roc_auc_cc = auc(fpr_cc, tpr_cc)
    
    plt.plot(fpr_ss, tpr_ss, 'b', label='Train: Synthea, Test: Synthea. AUC = %0.3f' % roc_auc_ss)
    plt.plot(fpr_sc, tpr_sc, 'g', label='Train: Synthea, Test: Cleveland. AUC = %0.3f' % roc_auc_sc)
    plt.plot(fpr_cc, tpr_cc, 'k', label='Train: Cleveland, Test: Cleveland. AUC = %0.3f' % roc_auc_cc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.plot([0.24, 0.24], [0,1], 'k--')
    plt.plot([0,1], [0.66,0.66], 'k--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
#    plt.savefig('roc_12_3_1542.png', format='png', dpi=1200)