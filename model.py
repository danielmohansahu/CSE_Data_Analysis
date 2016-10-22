# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:34:47 2016

@author: Jiachen
"""

#from sklearn.datasets import load_iris
#from sklearn.feature_selection import SelectFromModel
from sklearn import tree
import numpy as np

TREE_UNDEFINED = tree._tree.TREE_UNDEFINED

class model:
    
    def __init__(self, sourcefile, outfile):
        self.sourcefile = sourcefile
        self.outfile = outfile
            
        # Extract required information from raw csv
        self.num_point, self.data, self.target, self.feature_names = self.parse_data(sourcefile)
#        self.target_names = ['negative', 'positive']
        self.target_names = ['setosa', 'versicolor', 'virginica']
        
        # Construct decision tree
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(self.data, self.target)

        self.clf_tree = self.clf.tree_

    
    def parse_data(self, sourcefile):
        """
        Extract data, target (labels) and feature names (headers) from raw CSV
        """

        f = open(sourcefile, 'r')
        
        num_point = 0
        f.readline() # skip over header
        for line in f:
            num_point += 1
        f.seek(0) # rewind

        header_text = f.readline()
        list_feature_name = header_text.split(',')[0:-1]
        # Count number of commas (no need for +1) because the last column should be
        # the target
        num_feature = header_text.count(',')             
        f.seek(0) # rewind

        # initialize arrays
        target = np.zeros(num_point, dtype=int)
        data = np.zeros((num_point, num_feature))
        
        f.readline() # skip over header
        idx = 0
        for line in f:
            list_token = map(float, line.strip().split(','))
            target[idx] = int(list_token[-1])
            data[idx] = list_token[0:-1]
            idx += 1
        f.close()
        
        return num_point, data, target, list_feature_name


    def tree_to_code(self):
        """
        Traverses the decision tree and output representation of tree in 
        pseudocode (if...else...if...)
        Note that the left branch is always the branch that satisfies the 
        decision rule.
        """
        
        # Translate clf_tree.features from feature labels to feature strings
        feature_name = []
        for idx_feature in self.clf_tree.feature:
            if idx_feature != TREE_UNDEFINED:
                feature_name.append(self.feature_names[idx_feature])
            else:
                feature_name.append("undefined")
                            
        print "def tree():"
        
        def recurse(node, depth):
            indent = "    " * depth
            if self.clf_tree.feature[node] != TREE_UNDEFINED:
                # Not a leaf
                name = feature_name[node]
                threshold = self.clf_tree.threshold[node]
                # List of counts for each class
                values = self.clf_tree.value[node][0]
                total = np.sum(values)
                percentages = [ num*100.0/total for num in values]
                
                s = "{}# ".format(indent)
                for idx in range(0, len(self.target_names)):
                    s += "%s:%d%% " % (self.target_names[idx], percentages[idx])
                print s

                print "{}if {} <= {}:".format(indent, name, threshold)
                # Left branch satisfies rule
                recurse( self.clf_tree.children_left[node], depth+1 )
                
                print "{}else:".format(indent)
                # Right branch does not satisfy rule
                recurse( self.clf_tree.children_right[node], depth+1 )
            else:
                # At a leaf
                values = self.clf_tree.value[node][0]
                idx_feature = np.argmax(values)
                # Return the majority
                print "{}return {}".format(indent, self.target_names[idx_feature])
                
        recurse(0,1)
                

    def tree_to_json(self):
        """
        Writes tree to json output
        json output is a list of objects
        Each object has the following fields:
            node_id - unique id
            is_leaf - binary indicator
            feature_type - "categorical" or "numeric"
            feature_name - e.g. "cholesterol"
            val - only for categorical decision, e.g. "irish"
            threshold - only for numeric decision, e.g. "<= 140"
            num_negative - number of healthy patients at this node
            num_positive - number of patients with heart disease at this node
            label - "internal_node" or heart_disease" or "healthy"
            left - id of left child (the one that satisfies the rule)
            right - id of right child
        
        """
        tree_names = []
        for idx_feature in self.clf_tree.feature:
            if idx_feature != TREE_UNDEFINED:
                tree_names.append(self.feature_names[idx_feature])
            else:
                tree_names.append("undefined")
                            
        final_list = []
        
        def recurse(node, depth):
            node_id = node
            feature_name = tree_names[node]            
            if "categorical" in feature_name:
                feature_type = "categorical"
                # assumes that the categorical feature has been converted
                # into binary 1,0, so that the decision rule is simply
                # the proposition "is <name>"
                val = feature_name 
                threshold = -1
            else:
                feature_type = "numeric"
                val = ""
                threshold = self.clf_tree.threshold[node]
            counts = self.clf_tree.value[node][0]
            num_negative = counts[0]
            num_positive = counts[1]
            
            if self.clf_tree.feature[node] != TREE_UNDEFINED:
                # Not a leaf
                is_leaf = 0                
                label = "internal_node"
                recurse( self.clf_tree.children_left[node], depth+1 )
                recurse( self.clf_tree.children_right[node], depth+1 )
            else:
                is_leaf = 1
                if ( num_positive > num_negative ):
                    label = "heart_disease"
                else:
                    label = "healthy"

            node_map = {"node_id": node_id, "is_leaf": is_leaf,
                        "feature_name":feature_name, "feature_type":feature_type,
                        "val":val, "threshold":threshold, "num_negative":num_negative,
                        "num_positive":num_positive, "label":label,
                        "left":self.clf_tree.children_left[node],
                        "right":self.clf_tree.children_right[node]}
            final_list.append( node_map )
                
        recurse(0,1)
        return final_list            
    