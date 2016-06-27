%module nearest_neighbors_feature_selection

%{
    #define SWIG_FILE_WITH_INIT
    #include "nearest_neighbors_feature_selection.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* Xtrain, int nXtrain, int dXtrain)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *Ytrain, int nYtrain)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *selected_dimensions_extern, int num_selected_dimensions_extern)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *selected_ordering_extern, int num_selected_ordering_extern)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *selected_errors_extern, int num_selected_errors_extern)}

%include "nearest_neighbors_feature_selection.h"
