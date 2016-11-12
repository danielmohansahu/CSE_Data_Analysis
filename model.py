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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import random
import numpy as np
import json

TREE_UNDEFINED = tree._tree.TREE_UNDEFINED

class model:
    
    def __init__(self, trainfile, testfile, outfile='tree.json', out_python='tree.py'):
        self.trainfile = trainfile
        self.outfile = outfile
        self.out_python = out_python
            
        # Extract required information from raw csv
        # num_point - number of datapoints
        # data - numpy array of size num_point x num_feature
        # target - numpy array of size num_point x 1
        # feature_names - list of strings
        self.num_point, self.data, self.target, self.feature_names = self.read_data(trainfile)
        self.num_point_test, self.data_test, self.target_test, feature_names = self.read_data(testfile)
        self.target_names = ['negative', 'positive']
#        self.target_names = ['setosa', 'versicolor', 'virginica']
        

    def randomize_data(self, sourcefile, outfile):
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
            rand_idx = random.randint(idx, num_rows-1)
            # Swap
            list_lines[idx] = list_lines[rand_idx]
            list_lines[rand_idx] = temp
        
        with open(outfile, 'w') as f:
            f.writelines(list_lines)


    def read_data(self, sourcefile):
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

        model = SelectFromModel(clf, threshold='0.25*mean', prefit=True)
        # Eliminate variables that have been filtered out
        self.data = model.transform(self.data)
        self.data_test = model.transform(self.data_test)
        
        discarded_features = self.feature_names[model.get_support() == False]
        self.feature_names = self.feature_names[model.get_support()]
        
        print "Discarded features are:"
        print discarded_features        
        
        print "Remaining features are:"
        print self.feature_names


    def train(self):
        """
        Trains a single decision tree and outputs accuracy, precision, recall,
        and f1 score.
        Parameters of decision tree were acquired from prior runs of train_gridsearch()
        """
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=6,
                                          min_impurity_split=1e-5, min_samples_split=25,
                                          min_samples_leaf=10)
        clf = clf.fit(self.data, self.target)
        
        self.clf_tree = clf.tree_
        
        y_predicted = clf.predict(self.data_test)

        print "Mean accuracy"
        print clf.score(self.data_test, self.target_test)
        
        self.evaluate(self.target_test, y_predicted)
        

    def learning_curve(self, choice=0):
        
        if choice == 0:
            clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=6,
                                              min_impurity_split=1e-5, min_samples_split=25,
                                              min_samples_leaf=10)
        elif choice == 1:
            clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
        elif choice == 2:
            clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', max_depth=6,
                                              min_impurity_split=1e-5, min_samples_split=25,
                                              min_samples_leaf=10), n_estimators=10)
        train_sizes = range(500,1050,50)
        train_sizes, train_scores, valid_scores = learning_curve(clf, self.data,
                                                                 self.target, train_sizes=train_sizes,
                                                                 cv=5)
        print train_sizes
        print train_scores
        plt.plot(train_sizes, np.mean(train_scores, axis=1), c='red', ls='-')
        plt.plot(train_sizes, np.mean(valid_scores, axis=1), c='blue', ls='-')
        plt.show()
        

    def train_extratree(self):
        """
        Incomplete. To try next
        """

        clf = ExtraTreesClassifier(n_estimators=10, min_samples_split=2,
                                   random_state=0)
        clf = clf.fit(self.data, self.target)
                
        y_predicted = clf.predict(self.data_test)

        print "Mean accuracy"
        print clf.score(self.data_test, self.target_test)
        
        self.evaluate(self.target_test, y_predicted)


    def train_adaboost(self, choice=0):
        """
        Boost weak learner to produce strong learner
        """

        
        if choice == 0:
            clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', max_depth=6,
                                              min_impurity_split=1e-5, min_samples_split=25,
                                              min_samples_leaf=10), n_estimators=600)
            scores = cross_val_score(clf, self.data, self.target, cv=3)        
            return scores
        elif choice == 1:
            list_mean = []
            for n in range(100,650,50):
                clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', max_depth=6,
                                                  min_impurity_split=1e-5, min_samples_split=25,
                                                  min_samples_leaf=10), n_estimators=n)
                scores = cross_val_score(clf, self.data, self.target, cv=3)        
                print scores
                list_mean.append(scores.mean())
            return list_mean
        else:            
            clf = clf.fit(self.data, self.target)
    
            y_predicted = clf.predict(self.data_test)
    
            print "Mean accuracy"
            print clf.score(self.data_test, self.target_test)
            
            self.evaluate(self.target_test, y_predicted)               


    def train_gbc(self):
        """
        Incomplete. To try next
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target,
                                                            test_size=0.1, random_state=0)
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
                                         max_depth=2, random_state=0).fit(X_train, y_train)
                
        y_predicted = clf.predict(X_test)

        print "Mean accuracy"
        print clf.score(X_test, y_test)
        
        self.evaluate(y_test, y_predicted)        

        
    def evaluate(self, y_test, y_predicted):
        """
        Prints precision, recall, f1 score
        Inputs:
        1. y_test - the true labels
        2. y_predicted - labels predicted by model
        Obviously dimensions must match
        """
        
        print "f1_score"
        print metrics.f1_score(y_test, y_predicted, average=None)
        print "precision score"
        print metrics.precision_score(y_test, y_predicted)
        print "recall score"
        print metrics.recall_score(y_test, y_predicted)
        print "f1_score for positive class"
        print metrics.f1_score(y_test, y_predicted, average='binary')

        
    def train_gridsearch(self, verbose=0):
        """
        Exhaustive grid search with 5-fold cross validation to find good
        parameters for the decision tree        
        """
        
        parameters = {'max_depth': range(4,9), 'min_samples_leaf': range(10,50,10),
                      'min_impurity_split': list(np.logspace(-5,-2,4)),
                        'min_samples_split': range(15,35,5)}
                      
        # Searches over the given parameter grid to find the best set
        clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='gini'), 
                           parameters, cv=5)
        clf.fit(self.data, self.target)
        param_best = clf.best_params_
        print "Best parameters set found on development set:"
        print param_best 
        if verbose:
            print "Grid scores on development set:"
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))

        print "Mean accuracy"
        print clf.score(self.data_test, self.target_test)

        y_predicted = clf.predict(self.data_test)
        print "Scores using test set."        
        self.evaluate(self.target_test, y_predicted)                

        print "Classification report"
        print classification_report(self.target_test, y_predicted)

        # Use the "good" params to fit again
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=param_best['max_depth'],
                                          min_samples_leaf=param_best['min_samples_leaf'],
                                          min_impurity_split=param_best['min_impurity_split'],
                                            min_samples_split=param_best['min_samples_split'])
        clf = clf.fit(self.data, self.target)
        
        self.clf_tree = clf.tree_


    def train_gridsearch_extratree(self, verbose=0):
        """
        Exhaustive grid search with 5-fold cross validation to find good
        parameters for the decision tree        
        """
        
        parameters = {'min_impurity_split': list(np.logspace(-5,-2,4)),
                      'min_samples_split': range(5,30,5)}
                      
        # Searches over the given parameter grid to find the best set
        clf = GridSearchCV(ExtraTreesClassifier(n_estimators=10, max_depth=8,
                                                criterion='gini'), 
                           parameters, cv=5)
        clf.fit(self.data, self.target)
        param_best = clf.best_params_
        print "Best parameters set found on development set:"
        print param_best 
        if verbose:
            print "Grid scores on development set:"
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))

        print "Mean accuracy"
        print clf.score(self.data_test, self.target_test)

        y_predicted = clf.predict(self.data_test)
        print "Scores using test set."        
        self.evaluate(self.target_test, y_predicted)                

        print "Classification report"
        print classification_report(self.target_test, y_predicted)


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
#                    s += "%s:%d%% " % (self.target_names[idx], percentages[idx])
                    s += "%s:%d " % (self.target_names[idx], values[idx])                
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
#                    s += "%s:%d%% " % (self.target_names[idx], percentages[idx])
                    s += "%s:%d " % (self.target_names[idx], values[idx])     
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
