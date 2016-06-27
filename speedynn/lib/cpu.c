/* 
 * cpu.c
 */
#include "cpu.h"

// the current (global) distance matrix
FLOAT_TYPE *current_dist_matrix;
FLOAT_TYPE *dist_matrix_tmp;
FLOAT_TYPE *training_patterns_transpose;
FLOAT_TYPE *cross_validation_errors;


void init_memory_cpu(int large_scale){
	current_dist_matrix = (FLOAT_TYPE*)malloc(num_training_patterns*num_training_patterns*sizeof(FLOAT_TYPE));    
}

void free_memory_cpu(int large_scale){
	free(current_dist_matrix);
}

void get_smallest_validation_errors_cpu(int *selected_dimensions, FLOAT_TYPE *val_errors, int mult, int large_scale, int *loss_functions, int num_loss_functions, int model_type){
    int check_dim;
    if (mult == +1){
        check_dim = 0;
    } else {
        check_dim = 1;
    }
    // this array contains the cross validation errors (sum over the folds) for each k value
    FLOAT_TYPE *k_errors = (FLOAT_TYPE*)calloc(num_loss_functions*num_k_vals,sizeof(FLOAT_TYPE));    
    int current_dimension;
    // compute all validation errors
    for (current_dimension=0; current_dimension<dim; current_dimension++){
        // if the dimension has not been removed yet
        if (selected_dimensions[current_dimension] == check_dim){
            int i,j,m,l;
            
            // number of elements per fold
            int num_per_fold = (int) (num_training_patterns / k_fold);
            
            START_TIMER(3);

            FLOAT_TYPE *all_predictions = (FLOAT_TYPE*)malloc(num_training_patterns*num_k_vals*sizeof(FLOAT_TYPE));

            if (model_type==REGRESSION_MODEL){            
                // for each training pattern
                for (i=0;i<num_training_patterns;i++){
                    // compute fold of training pattern
                    int fold_of_pattern = (int) i / num_per_fold;
                    // compute the kmax nearest neighbors in the remaining folds
                    int *k_max_nearest_neighbors_indices = (int*)malloc(k_max*sizeof(int));
                    FLOAT_TYPE *k_max_nearest_neighbors_distances = (FLOAT_TYPE*)malloc(k_max*sizeof(FLOAT_TYPE));
                    get_k_nearest_neighbors_cpu(i, fold_of_pattern, num_per_fold, \
                                        k_max_nearest_neighbors_indices, k_max_nearest_neighbors_distances, \
                                        current_dimension, mult);
                    // for each k value 
                    for (m=0; m<num_k_vals; m++){
                        int k=k_vals[m];
                        // compute predictions for each k value based on the k nearest neighbors
                        FLOAT_TYPE pred = 0.0;
                        for (j=0; j<k; j++){
                            // the k nearest neighbors are stored in the first k slots of k_max_nearest_neighbors_indices
                            pred += training_labels[k_max_nearest_neighbors_indices[j]];
                        }
                        pred /= k;
                        all_predictions[m*num_training_patterns + i] = pred;
                    }
                    free(k_max_nearest_neighbors_indices);
                    free(k_max_nearest_neighbors_distances);
                }
            } else if (model_type==CLASSIFICATION_MODEL) {
                // for each training pattern
                for (i=0;i<num_training_patterns;i++){
                    // compute fold of training pattern
                    int fold_of_pattern = (int) i / num_per_fold;
                    // compute the kmax nearest neighbors in the remaining folds
                    int *k_max_nearest_neighbors_indices = (int*)malloc(k_max*sizeof(int));
                    FLOAT_TYPE *k_max_nearest_neighbors_distances = (FLOAT_TYPE*)malloc(k_max*sizeof(FLOAT_TYPE));
                    get_k_nearest_neighbors_cpu(i, fold_of_pattern, num_per_fold, \
                                        k_max_nearest_neighbors_indices, k_max_nearest_neighbors_distances, \
                                        current_dimension, mult);
                    // for each k value 
                    for (m=0; m<num_k_vals; m++){
                        int k=k_vals[m];
                        // compute predictions for each k value based on the k nearest neighbors
                        FLOAT_TYPE *all_preds = (FLOAT_TYPE*)calloc(k,sizeof(FLOAT_TYPE));
                        for (j=0; j<k; j++){
                            // the k nearest neighbors are stored in the first k slots of k_max_nearest_neighbors_indices
                            all_preds[j] = training_labels[k_max_nearest_neighbors_indices[j]];
                        }
                        all_predictions[m*num_training_patterns + i] = get_mode_array(all_preds, k);
                        free(all_preds);
                    }
                    free(k_max_nearest_neighbors_indices);                    
                    free(k_max_nearest_neighbors_distances);
                }
            
            } else {
                printf("model type not implemented...\n");
                exit(0);
            }
            STOP_TIMER(3);
            RECORD_TIMER_SUM(3);

            for (l=0;l<num_loss_functions;l++){
                for (m=0; m<num_k_vals; m++){
                    k_errors[l*num_k_vals + m] = get_loss(training_labels, num_training_patterns, all_predictions+m*num_training_patterns, num_training_patterns, loss_functions[l]);
                }
            }

            // compute the smallest (==best) validation error
            FLOAT_TYPE smallest_validation_error = MAX_FLOAT_TYPE;
            for (m=0; m<num_k_vals; m++){
                if (k_errors[m] < smallest_validation_error){
                    smallest_validation_error = k_errors[m];
                    best_current_k_model_parameters[current_dimension] = k_vals[m];
                    for (l=0;l<num_loss_functions;l++){
                        val_errors[l*dim + current_dimension] = k_errors[l*num_k_vals + m];
                    }
                }
            }

            free(all_predictions);
        }
    }
    free(k_errors);
}

void get_k_nearest_neighbors_cpu(int train_idx, int fold, int num_per_fold, int *k_max_nn_indices, FLOAT_TYPE *k_max_nn_distances, int current_dimension, int mult){
    int i,j,m;
    // init arrays
    for (j=0;j<k_max;j++){
        k_max_nn_indices[j] = 0;
        k_max_nn_distances[j] = MAX_FLOAT_TYPE;
    }
    FLOAT_TYPE tmp;
    int tmp_idx;
    // traverse all nearest neighbors and update the kmax nn indices accordingly
    for (i=0;i<num_training_patterns;i++){
        if (i / num_per_fold != fold){
            // ingore all values that are within the same fold
            FLOAT_TYPE d;
            if (current_dimension >= 0){
                 d = current_dist_matrix[train_idx*num_training_patterns + i] + \
                            mult*dist_dimension(training_patterns+train_idx*dim, training_patterns+i*dim, current_dimension);
            } else {
                 d = current_dist_matrix[train_idx*num_training_patterns + i];
            }
            // insert dist/idx in local arrays
            j=k_max-1;	    
            if(k_max_nn_distances[j]>d){
                k_max_nn_distances[j]=d;
                k_max_nn_indices[j]=i;    
                for(;j>0;j--) {
                    if(k_max_nn_distances[j]<k_max_nn_distances[j-1]){
	                    //swap dist
	                    tmp=k_max_nn_distances[j];
	                    k_max_nn_distances[j]=k_max_nn_distances[j-1];
	                    k_max_nn_distances[j-1]=tmp;
	                    //swap idx
	                    tmp_idx=k_max_nn_indices[j];
	                    k_max_nn_indices[j]=k_max_nn_indices[j-1];
	                    k_max_nn_indices[j-1]=tmp_idx;
                    } else break;
                }
            }
        }
    }
}


void empty_distance_matrix_cpu(){
    START_TIMER(2);
    int i,j;
    // for all (training) patterns
    for (i=0; i<num_training_patterns; i++){
        // symmetric distance matrix
        for (j=0; j<num_training_patterns; j++){
            current_dist_matrix[i*num_training_patterns + j] = 0.0;
        }
    }
    STOP_TIMER(2);
    RECORD_TIMER_SUM(2);	
}

void compute_distance_matrix_cpu(int *selected_dimensions, int selected_only){
    START_TIMER(2);
    int i,j;
    // for all (training) patterns
    for (i=0; i<num_training_patterns; i++){
        // we only have 0 values on the diagonal
        current_dist_matrix[i*num_training_patterns + i] = 0.0;
        // symmetric distance matrix
        for (j=i+1; j<num_training_patterns; j++){
            FLOAT_TYPE d = 0.0;            
            if (!selected_only){
                // compute distance based on all dimensions            
                d = dist(training_patterns+i*dim, training_patterns+j*dim, dim);
            } else {
                int l;
                // compute distance based on selected dimensions only
                for (l=0; l<dim; l++){
                    if (selected_dimensions[l] == 1){
                        d += dist_dimension(training_patterns+i*dim, training_patterns+j*dim, l);
                    }
                }            
            }
            // symmetric matrix
            current_dist_matrix[i*num_training_patterns + j] = d;
            // not needed? remove?
            current_dist_matrix[j*num_training_patterns + i] = d;
        }
    }
    STOP_TIMER(2);
    RECORD_TIMER_SUM(2);	    
}

void update_current_dist_matrix_cpu(int min_dim, int mult){
    START_TIMER(4);
    int i,j;
    for (i=0; i<num_training_patterns; i++){
        // we only have zero values on the diagonal...
        for (j=i+1; j<num_training_patterns; j++){
            // NOTE: Only O(1) work here        
            FLOAT_TYPE d = dist_dimension(training_patterns+i*dim, training_patterns+j*dim, min_dim);
            current_dist_matrix[i*num_training_patterns + j] = current_dist_matrix[i*num_training_patterns + j] + mult*d;
            current_dist_matrix[j*num_training_patterns + i] = current_dist_matrix[j*num_training_patterns + i] + mult*d;
        }
    }
    STOP_TIMER(4);
    RECORD_TIMER_SUM(4);
}

