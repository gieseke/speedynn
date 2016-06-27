/* 
 * extern.h
 */
#ifndef EXTERN
#define EXTERN

#include "types.h"
#include "util.h"
#include "nearest_neighbors_feature_selection.h"

// Definition of extern variables

// dimension of patterns
extern int dim;
// number of nearest neighbors (and range/max value)
extern int *k_vals;
extern int num_k_vals;
extern int k_max;
extern int *best_current_k_model_parameters;
// number of desired features
extern int num_desired_features;
// selected dimensions (1 indicates that dimension is selected, 0 not)
extern int *selected_dimensions;
// k-fold cross validation
extern int k_fold;
// the training patterns
extern FLOAT_TYPE *training_patterns;
// number of training patterns
extern int num_training_patterns;
// the training labels (needed for cross-validation)
extern FLOAT_TYPE *training_labels;
// the number of training labels
extern int num_training_labels;




DECLARE_TIMER(1);
DECLARE_TIMER(2);
DECLARE_TIMER(3);
DECLARE_TIMER(4);
DECLARE_TIMER(5);
DECLARE_TIMER(6);

#endif

