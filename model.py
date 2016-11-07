# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:34:47 2016

@author: Jiachen
"""

#from sklearn.datasets import load_iris
#from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import numpy as np
import json

TREE_UNDEFINED = tree._tree.TREE_UNDEFINED

class model:
    
    def __init__(self, sourcefile, outfile, out_python='tree.py'):
        self.sourcefile = sourcefile
        self.outfile = outfile
        self.out_python = out_python
            
        # Extract required information from raw csv
        # num_point - number of datapoints
        # data - numpy array of size num_point x num_feature
        # target - numpy array of size num_point x 1
        # feature_names - list of strings
        self.num_point, self.data, self.target, self.feature_names = self.parse_data(sourcefile)
        self.target_names = ['negative', 'positive']
#        self.target_names = ['setosa', 'versicolor', 'virginica']
        


    def parse_data(self, sourcefile):
        """
        Extract data, target (labels) and feature names (headers) from raw CSV
        """

        f = open(sourcefile, 'r')
        
        num_point = 0 # number of datapoints
        f.readline() # skip over header
        for line in f:
            num_point += 1
        f.seek(0) # rewind

        header_text = f.readline()
        feature_names = np.array(header_text.split(',')[0:-1])
        # Count number of commas (no need for +1) because the last column should be
        # the target
        num_feature = header_text.count(',')
        f.seek(0) # rewind

        # initialize arrays
        target = np.zeros(num_point, dtype=int)
        data = np.zeros((num_point, num_feature), dtype=np.float32)
        
        f.readline() # skip over header
        idx = 0
        for line in f:
            list_token = map(float, line.strip().split(','))
            target[idx] = int(list_token[-1])
            data[idx] = list_token[0:-1]
            idx += 1
        f.close()
        
        return num_point, data, target, feature_names


    def pre_selection(self):
        """
        Uses extra-trees classifier to estimate the importance of features.
        Eliminates all features whose importance is below the mean of all values
        The extra-trees method fits many randomized decision trees on random 
        subsamples of the data and uses averaging to determine the importance
        of a feature.
        """
        clf = ExtraTreesClassifier(n_estimators=10, criterion="gini")
        clf = clf.fit(self.data, self.target)

        print "Feature importances:"
        print clf.feature_importances_

        model = SelectFromModel(clf, threshold='mean', prefit=True)
        # Eliminate variables that have been filtered out
        self.data = model.transform(self.data)
        
        discarded_features = self.feature_names[model.get_support() == False]
        self.feature_names = self.feature_names[model.get_support()]
        
        print "Discarded features are:"
        print discarded_features        
        
        print "Remaining features are:"
        print self.feature_names


    def train(self):
        clf = tree.DecisionTreeClassifier().fit(self.data, self.target)
        self.clf_tree = clf.tree_


    def train_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target,
                                                            test_size=0.1, random_state=0)        
        clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
        print clf.score(X_test, y_test)
        
        
    def train_gridsearch(self):
        # Construct decision tree
        """
        Parameters to tune: criterion, max_depth, min_impurity_split
        """
    
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target,
                                                            test_size=0.25, random_state=0)
        
        parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(5,10),
                      'min_impurity_split': list(np.logspace(-8,-2,7))}
                      
        # Searches over the given parameter grid to find the best set
        clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)
        clf.fit(X_train, y_train)
        param_best = clf.best_params_
        print "Best parameters set found on development set:"
        print param_best 
        print "Grid scores on development set:"
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        
        print "Detailed classification report:"
        print"The model is trained on the full development set."
        print"The scores are computed on the full evaluation set."
        y_true, y_pred = y_test, clf.predict(X_test)
        print classification_report(y_true, y_pred)

        # Use the best params to fit again
        clf = tree.DecisionTreeClassifier(criterion=param_best['criterion'],
                                          min_impurity_split=param_best['min_impurity_split'],
                                            max_depth=param_best['max_depth'])
        clf = clf.fit(self.data, self.target)
        self.clf_tree = clf.tree_


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
    
        f = open(self.out_python, 'w')
        f.write('def tree():\n')
        
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
                f.write(s+"\n")

                f.write("{}if {} <= {}:\n".format(indent, name, threshold))
                # Left branch satisfies rule
                recurse( self.clf_tree.children_left[node], depth+1 )
                
                f.write("{}else:\n".format(indent))
                # Right branch does not satisfy rule
                recurse( self.clf_tree.children_right[node], depth+1 )
            else:
                # At a leaf
                values = self.clf_tree.value[node][0]
                total = np.sum(values)
                percentages = [ num*100.0/total for num in values]                
                idx_feature = np.argmax(values)
                # Return the majority
                s = "{}# ".format(indent)
                for idx in range(0, len(self.target_names)):
                    s += "%s:%d%% " % (self.target_names[idx], percentages[idx])
                f.write(s+"\n")             
                f.write("{}return {}\n".format(indent, self.target_names[idx_feature]))
                
        recurse(0,1)
        f.close()


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
        # clf_tree.feature is a list of indices, e.g. [0, 3, -2, 1, ... ]
        # Each index of the list corresponds to a node in the tree
        # -2 means the node is a leaf and hence does not have a feature
        # Numbers other than -2 are indices of self.feature_names,
        # indicating which feature is used for splitting at a node
        for idx_feature in self.clf_tree.feature:
            if idx_feature != TREE_UNDEFINED:
                # Get the actual feature name
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
                # clf_tree.threshold is a numpy array of real numbers
                # Value v_n at index n is the value used for the decision rule at node n
                # if (x < threshold) then branch left else branch right
                threshold = self.clf_tree.threshold[node]
            # clf_tree.value is a 3D numpy matrix
            # Topmost dimension picks out rows. Each row is a node.
            # Second dimension appears useless. Last dimension goes across a row.
            # Value v_n in a row is the number of datapoints belonging to the nth label
            # that ended up in that node.
            counts = self.clf_tree.value[node][0]
            num_negative = counts[0]
            num_positive = counts[1]
            
            if self.clf_tree.feature[node] != TREE_UNDEFINED:
                # Not a leaf
                is_leaf = 0                
                label = "internal_node"
                # clf_tree.children_left is a numpy array, where each index is a node
                # and the value v_n at index n is the index of the left child of node n
                # Values at indices corresponding to leaf nodes have value -1.
                # Same for clf_tree.children_right
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
        
        with open(self.outfile, 'w') as f:
            json.dump(final_list, f, indent=4, sort_keys=True, separators=(',', ':'))

        return final_list


    def tree_to_json2(self):
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
        # clf_tree.feature is a list of indices, e.g. [0, 3, -2, 1, ... ]
        # Each index of the list corresponds to a node in the tree
        # -2 means the node is a leaf and hence does not have a feature
        # Numbers other than -2 are indices of self.feature_names,
        # indicating which feature is used for splitting at a node
        for idx_feature in self.clf_tree.feature:
            if idx_feature != TREE_UNDEFINED:
                # Get the actual feature name
                tree_names.append(self.feature_names[idx_feature])
            else:
                tree_names.append("undefined")
                            
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
                # clf_tree.threshold is a numpy array of real numbers
                # Value v_n at index n is the value used for the decision rule at node n
                # if (x < threshold) then branch left else branch right
                threshold = self.clf_tree.threshold[node]
            # clf_tree.value is a 3D numpy matrix
            # Topmost dimension picks out rows. Each row is a node.
            # Second dimension appears useless. Last dimension goes across a row.
            # Value v_n in a row is the number of datapoints belonging to the nth label
            # that ended up in that node.
            counts = self.clf_tree.value[node][0]
            num_negative = counts[0]
            num_positive = counts[1]
            
            if self.clf_tree.feature[node] != TREE_UNDEFINED:
                # Not a leaf
                is_leaf = 0                
                label = "internal_node"
                map_internal = {}
                if feature_type == "categorical":
                    map_internal["name"] = "%s == %s" % (feature_name, val)
                else:
                    map_internal["name"] = "%s < %.3f" % (feature_name, threshold)
                map_internal["node_id"] = node_id
                map_internal["is_leaf"] = is_leaf
                map_internal["num_negative"] = num_negative
                map_internal["num_positive"] = num_positive
                map_internal["label"] = label
                children = []
                # clf_tree.children_left is a numpy array, where each index is a node
                # and the value v_n at index n is the index of the left child of node n
                # Values at indices corresponding to leaf nodes have value -1.
                # Same for clf_tree.children_right
                children.append( recurse( self.clf_tree.children_left[node], depth+1 ) )
                children.append( recurse( self.clf_tree.children_right[node], depth+1 ) )
                map_internal["children"] = children
                return map_internal
            else:
                is_leaf = 1
                if ( num_positive > num_negative ):
                    label = "heart_disease"
                else:
                    label = "healthy"
                map_leaf = {"name": label, "node_id": node_id, "is_leaf": is_leaf,
                            "num_positive": num_positive, "num_negative": num_negative,
                            "label": label}
                return map_leaf
                
        final_map = recurse(0,1)
        
        with open(self.outfile, 'w') as f:
            json.dump(final_map, f, indent=4, sort_keys=True, separators=(',', ':'))

        return final_map
