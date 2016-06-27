import sys
import numpy
import generate

from speedynn.base import IncrementalNNFeatureSelection
from sklearn.datasets import make_regression

# parameters
use_gpu = 1
method = "forward"
n_neighbors_range = range(2,51)
num_features = 3
k_fold = 10
loss_functions = ["rmse"]
auto_feature = False

# generate artificial data
Xtrain, Ytrain = make_regression(n_samples=100, n_features=10, n_informative=3, noise=0.5, coef=False, bias=0.0, random_state=0)

# apply feature selection scheme
knn = IncrementalNNFeatureSelection(Xtrain, 
                                    Ytrain, 
                                    k_vals=n_neighbors_range, 
                                    num_features=num_features, \
                                    k_fold=k_fold, \
                                    method=method, 
                                    loss_functions=loss_functions, 
                                    auto_feature=auto_feature, 
                                    use_gpu=use_gpu
                                    )
knn.train()

print("Selected dimensions: %s" % str(knn.get_selected_dimensions()))
print("Training errors: %s" % str(knn.get_selected_errors()))
