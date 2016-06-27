/* 
 * nearest_neighbors_feature_selection.h
 */
#ifndef NEAREST_NEIGHBORS_FEATURE_SELECTION
#define NEAREST_NEIGHBORS_FEATURE_SELECTION

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <sched.h>


#include "types.h"
#include "util.h"
#include "extern.h"
#include "cpu.h"
#include "gpu.h"

#define REGRESSION_MODEL 0
#define CLASSIFICATION_MODEL 1

#if DEBUG > 0
    #define DEBUG_PRINT printf
#else
    #define DEBUG_PRINT
#endif

#if USE_GPU > 0 
    #define INIT_OPENCL_DEVICES() init_opencl_devices()
    #define FREE_OPENCL_DEVICES() free_opencl_devices()  
    #define INIT_MEMORY init_memory_gpu
    #define FREE_MEMORY free_memory_gpu
    #define COMPUTE_DISTANCE_MATRIX compute_distance_matrix_gpu
    #define EMPTY_DISTANCE_MATRIX empty_distance_matrix_gpu                
    #define UPDATE_CURRENT_DIST_MATRIX update_current_dist_matrix_gpu
    #define GET_SMALLEST_VALIDATION_ERRORS get_smallest_validation_errors_gpu
    #define GET_VALIDATION_ERROR get_validation_error_gpu        
#else
    #define INIT_OPENCL_DEVICES() 
    #define FREE_OPENCL_DEVICES()    
    #define INIT_MEMORY init_memory_cpu
    #define FREE_MEMORY free_memory_cpu
    #define COMPUTE_DISTANCE_MATRIX compute_distance_matrix_cpu
    #define EMPTY_DISTANCE_MATRIX empty_distance_matrix_cpu            
    #define UPDATE_CURRENT_DIST_MATRIX update_current_dist_matrix_cpu
    #define GET_SMALLEST_VALIDATION_ERRORS get_smallest_validation_errors_cpu
    #define GET_VALIDATION_ERROR get_validation_error_cpu    
#endif


/* -------------------------------------------------------------------------------- 
 * Train: Backward Selection
 * -------------------------------------------------------------------------------- 
*/
int train_backward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, int *kvals, int num_kvals, int num_features, int kfold, int *loss_functions, int num_loss_functions, int auto_feature, int model_type);

/* -------------------------------------------------------------------------------- 
 * Train: Forward Selection
 * -------------------------------------------------------------------------------- 
*/
int train_forward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, int *kvals, int num_kvals, int num_features, int kfold, int *loss_functions, int num_loss_functions, int auto_feature, int model_type);
int backward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, int *kvals, int num_kvals, int num_features, int kfold, int large_scale, int *loss_functions, int num_loss_functions, int auto_feature, int model_type);
int forward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, int *kvals, int num_kvals, int num_features, int kfold, int large_scale, int *loss_functions, int num_loss_functions, int auto_feature, int model_type);

/* -------------------------------------------------------------------------------- 
 * Get selected dimensions
 * -------------------------------------------------------------------------------- 
*/
void get_selected_dimensions(int *selected_dimensions_extern, int num_selected_dimensions_extern);
void get_selected_ordering(int *selected_ordering_extern, int num_selected_ordering_extern);
void get_selected_errors(FLOAT_TYPE *selected_errors_extern, int num_selected_errors_extern);

int get_final_optimal_model_k();
FLOAT_TYPE get_loss_all(FLOAT_TYPE *labels, int num_labels, FLOAT_TYPE *predictions, int num_predictions, int loss_function);

int optimal_performance(FLOAT_TYPE *errors, int iter, int mult);

#endif


