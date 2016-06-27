%module nearest_neighbors_feature_selection

%{
    #define SWIG_FILE_WITH_INIT
    #include "nearest_neighbors_feature_selection.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* Xtrain, int nXtrain, int dXtrain)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *Ytrain, int nYtrain)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *kvals, int num_kvals)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *selected_dimensions_extern, int num_selected_dimensions_extern)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *selected_ordering_extern, int num_selected_ordering_extern)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *selected_errors_extern, int num_selected_errors_extern)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *labels, int num_labels)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *predictions, int num_predictions)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *loss_functions, int num_loss_functions)}

%include "nearest_neighbors_feature_selection.h"
