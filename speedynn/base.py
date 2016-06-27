import numpy as np
import time
import os
import sys
import copy
import random
import math

class IncrementalNNFeatureSelection:
    """
    Efficient implementation for nearest neighbor feature selection
    """
    loss_dict = {
                "rmse":0,
                "rmse_norm":1,
                "mad":2,
                "mad_norm":3,
                "zero_one":4,
                "std":5,
                "rmse_norm_mad_norm_std":6,
                "rmse_norm_mad_norm_galaxies":7,
                "rmse_std":8
                }
    model_types = {
                "regression":0,
                "classification":1
                }                
                
    def __init__(self, Xtrain, Ytrain, k_vals=None, auto_feature=False, num_features=2, k_fold=10, method="backward", loss_functions=["rmse"], model_type="regression", use_gpu=True):
        self.Xtrain = np.array(Xtrain)
        self.Ytrain = Ytrain
        self.dim = len(Xtrain[0])
        self.auto_feature = int(auto_feature)

        if k_vals == None:
            self.k_vals = np.array(range(1,10))
        else:
            self.k_vals = np.array(k_vals)
        self.num_desired_features = num_features
        assert self.num_desired_features <= self.dim
        assert self.num_desired_features > 0
        if auto_feature:
            self.num_desired_features = self.dim
        # convert arrays to correct types    
        self.dtype_int = np.int32        
        self.dtype_float = np.float32   
        self.Xtrain = self.Xtrain.astype(self.dtype_float)
        self.Ytrain = self.Ytrain.astype(self.dtype_float)
        self.k_vals = self.k_vals.astype(self.dtype_int)
        self.k_fold = k_fold
        self.method = method
        self.loss_functions = []
        for i in xrange(len(loss_functions)):
            self.loss_functions.append(self.loss_dict[loss_functions[i]])
        self.loss_functions = np.array(self.loss_functions).astype(self.dtype_int)
        self.model_type = self.model_types[model_type]
        self.use_gpu = use_gpu
        print "loss_functions: ", str(loss_functions)

    def train(self):
        if self.use_gpu:
            import nearest_neighbors_feature_selection_gpu as nearest_neighbors_feature_selection
        else:
            import nearest_neighbors_feature_selection_cpu as nearest_neighbors_feature_selection
        if self.method == "backward":
            self.num_final_features = nearest_neighbors_feature_selection.train_backward_selection(self.Xtrain, self.Ytrain, \
                                self.k_vals, self.num_desired_features, self.k_fold, self.loss_functions, self.auto_feature, self.model_type)
        elif self.method == "forward":
            self.num_final_features = nearest_neighbors_feature_selection.train_forward_selection(self.Xtrain, self.Ytrain, \
                                self.k_vals, self.num_desired_features, self.k_fold, self.loss_functions, self.auto_feature, self.model_type)  
        else:
            print "Error: Unknown feature selection method. Exiting ..."
            sys.exit(0)
        self.selected_dimensions = np.zeros(self.dim, dtype=self.dtype_int)
        nearest_neighbors_feature_selection.get_selected_dimensions(self.selected_dimensions)
        self.selected_ordering = np.zeros(self.dim, dtype=self.dtype_int)
        nearest_neighbors_feature_selection.get_selected_ordering(self.selected_ordering)
        
        self.selected_errors = np.zeros(self.dim*self.loss_functions.shape[0], dtype=self.dtype_float)
        nearest_neighbors_feature_selection.get_selected_errors(self.selected_errors)
        self.selected_errors_new = np.zeros(self.dim*self.loss_functions.shape[0], dtype=self.dtype_float).reshape((self.dim,self.loss_functions.shape[0]))
        for l in xrange(self.loss_functions.shape[0]):
            self.selected_errors_new[:,l] = self.selected_errors[l*self.dim:(l+1)*self.dim]
        self.selected_errors = self.selected_errors_new
        self.optimal_model_k = nearest_neighbors_feature_selection.get_final_optimal_model_k()

    def get_selected_dimensions(self):
        return self.selected_dimensions
        
    def get_selected_ordering(self):
        return self.selected_ordering

    def get_feature_ranks(self):
        if self.method == "backward":
            ranks = np.zeros(self.dim)
            for i in xrange(self.dim-self.num_final_features):
                ranks[self.selected_ordering[self.dim-i-1]] = - (self.dim-self.num_final_features) + i
            return ranks
        elif self.method == "forward":
            ranks = np.zeros(self.dim)
            for i in xrange(self.num_final_features):
                ranks[self.selected_ordering[i]] = self.num_final_features - i
            return ranks
        else:
            print "Error: Unknown feature selection method. Exiting ..."
            sys.exit(0)        
        return ranks

    def get_selected_errors(self):
        return self.selected_errors

    def get_optimal_model_k(self):
        return self.optimal_model_k

    def get_loss(self, labels, predictions, loss_func="rmse"):
        if self.use_gpu:
            import nearest_neighbors_feature_selection_gpu as nearest_neighbors_feature_selection
        else:
            import nearest_neighbors_feature_selection_cpu as nearest_neighbors_feature_selection
        loss_func = self.loss_dict[loss_func]
        return nearest_neighbors_feature_selection.get_loss_all(labels.astype(self.dtype_float), predictions.astype(self.dtype_float), loss_func)


