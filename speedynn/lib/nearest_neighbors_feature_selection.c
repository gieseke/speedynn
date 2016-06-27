/* 
 * nearest_neighbors_feature_selection.c
 */
#include "nearest_neighbors_feature_selection.h"

// dimension of patterns
int dim;
// number of nearest neighbors (and range/max value)
int *k_vals;
int num_k_vals;
int k_max;
int best_k_model_parameter;
int *best_current_k_model_parameters;
// number of desired features
int num_desired_features;
// selected dimensions (1 indicates that dimension is selected, 0 not)
int *selected_dimensions;
// the ordering in which the features are selected
int *selected_ordering;
// the corresponding errors
FLOAT_TYPE *selected_errors;
// k-fold cross validation
int k_fold;
// the training patterns
FLOAT_TYPE *training_patterns;
// number of training patterns
int num_training_patterns;
// the training labels (needed for cross-validation)
FLOAT_TYPE *training_labels;
// the number of training labels
int num_training_labels;
// the number of final features
int num_final_features;
// threshold for large-scale settings
int large_scale_threshold = 15000;

// timers used for debugging and runtime measurements
DEFINE_TIMER(1);
DEFINE_TIMER(2);
DEFINE_TIMER(3);
DEFINE_TIMER(4);
DEFINE_TIMER(5); 
DEFINE_TIMER(6); 

/* -------------------------------------------------------------------------------- 
 * Train: Backward Selection (SELECTOR)
 * -------------------------------------------------------------------------------- 
*/
int train_backward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, 
                             int *kvals, int num_kvals, int num_features, int kfold, int *loss_functions, 
                             int num_loss_functions, int auto_feature, int model_type) {
    int large_scale = (nXtrain > large_scale_threshold);
    return backward_selection(Xtrain, nXtrain, dXtrain, Ytrain, nYtrain, kvals, num_kvals, num_features, kfold, large_scale, loss_functions, num_loss_functions, auto_feature, model_type);
}

/* -------------------------------------------------------------------------------- 
 * Train: Forward Selection (SELECTOR)
 * -------------------------------------------------------------------------------- 
*/
int train_forward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, 
                            int *kvals, int num_kvals, int num_features, int kfold, int *loss_functions, 
                            int num_loss_functions,  int auto_feature, int model_type) {
    int large_scale = (nXtrain > large_scale_threshold);
    return forward_selection(Xtrain, nXtrain, dXtrain, Ytrain, nYtrain, kvals, num_kvals, num_features, kfold, large_scale, loss_functions, num_loss_functions, auto_feature, model_type);
}

/* ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------------- 
 * Train: Backward Selection (MEDIUM)
 * -------------------------------------------------------------------------------- 
*/
int backward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, 
                       int *kvals, int num_kvals, int num_features, int kfold, int large_scale, 
                       int *loss_functions, int num_loss_functions, int auto_feature, int model_type) {
	DEBUG_PRINT("-------------------------------------------\n");
	DEBUG_PRINT("Backward Feature Selection (LARGE_SCALE=%i, DEBUG=%i, USE_GPU=%i) \n", large_scale, DEBUG, USE_GPU);
	DEBUG_PRINT("-------------------------------------------\n");

    // large-scale is not implemented yet ...
    if (large_scale){
        printf("Not yet implemented. Exiting ...\n");
        exit(0);
    }

    // counters
	int i,j,l;	

    // set all parameters
	dim = dXtrain;
	training_patterns = Xtrain;	
	num_training_patterns = nXtrain;
	training_labels = Ytrain;
	num_training_labels = nYtrain;
	num_desired_features = num_features;
	// the final number of features (is the same as num_desired_features at the end if auto search is deactivated)
    num_final_features = dim;
	DEBUG_PRINT("Number of desired features=%i\n",num_desired_features);
    // automatic search for optimal feature dimension
    if (auto_feature){
    	DEBUG_PRINT("Automatic search for optimal feature dimension active!\n");
        num_desired_features = 0;
    }
    k_fold = kfold;
    DEBUG_PRINT("Number of folds=%i\n",k_fold);
	k_vals = kvals;
	num_k_vals = num_kvals;
	k_max = get_max_value(k_vals, num_k_vals);
    DEBUG_PRINT("Number of maximum nearest neighbors=%i\n",k_max);
    best_current_k_model_parameters = (int*)malloc(dim*sizeof(int));

    // init the OpenCl device (or do nothing in case USE_GPU==0)
    INIT_OPENCL_DEVICES();

    START_TIMER(1);
    // initialize all buffers and arrays
    INIT_MEMORY(large_scale);
    
	// initialize all dimensions that are selected (1=select, 0=not selected)
	selected_dimensions = (int*)malloc(dim*sizeof(int));
	for (i=0;i<dim;i++){selected_dimensions[i] = 1;}
    // initialize the ordering array (which features are selected first)
	selected_ordering = (int*)malloc(dim*sizeof(int));
	for (i=0;i<dim;i++){selected_ordering[i] = 0;}
    // initialize the error array (which contains, for each loss function specified,
    // the corresponding cross-validation error for each step)
	selected_errors = (FLOAT_TYPE*)malloc(num_loss_functions*dim*sizeof(FLOAT_TYPE));
	for (i=0;i<num_loss_functions*dim;i++){selected_errors[i] = 0.0;}
	
    // compute initial (complete) distance matrix if needed
    if (!large_scale){
	    COMPUTE_DISTANCE_MATRIX(NULL, 0);
	    DEBUG_PRINT("Distance matrix computed ...\n\n");
    }
    
    // array containing the intermediate validation errors for each tested dimension
    FLOAT_TYPE *val_errors = (FLOAT_TYPE*)malloc(num_loss_functions*dim*sizeof(FLOAT_TYPE));

	// Perform iterative feature selection: In each round, we identify 
    // (and remove) a single dimension that does carry the least useful information
	for (i=dim-1; i>= num_desired_features; i--){

	    DEBUG_PRINT("Processing iteration %i (of %i)\n",-(i-(dim-1))+1,dim-num_desired_features);
        num_final_features--;
        FLOAT_TYPE min_val_error = MAX_FLOAT_TYPE;
        int min_dim = -1;
        
        // compute all validation errors 
        GET_SMALLEST_VALIDATION_ERRORS(selected_dimensions, val_errors, -1, large_scale, loss_functions, num_loss_functions, model_type);
        
	    // now get the smallest val error
	    for (j=0; j<dim; j++){
	        // if the dimension has not been removed yet
	        if (selected_dimensions[j] == 1){
	            // we want to remove the dimension that, in case it is removed, leads to the best result
	            // (e.g., given three dimensions, and the last one is only noise, we want to remove it)
	            DEBUG_PRINT(" -  Dimensions %i has val_error=%f\n", j, val_errors[j]);
	            // we use the first loss function for selecting the best dimension
	            if (val_errors[j] < min_val_error){
	                min_val_error = val_errors[j];
                    min_dim = j;
	            }	        
            }
        }	    

        // the last dimension that is removed -> store corresponding optimal model parameter
        best_k_model_parameter = best_current_k_model_parameters[min_dim];

	    // remove dimension max_dim that corresponds to the maximal valiation error
	    DEBUG_PRINT(" -> Removing dimension %i!\n\n", min_dim);
	    selected_dimensions[min_dim] = 0;
	    selected_ordering[i] = min_dim;


	    // save all losses	    
	    for (l=0;l<num_loss_functions;l++){
    	    selected_errors[l*dim + i] = val_errors[l*dim + min_dim];
	    }

	    // update global distance matrix (we only need to change O(1 n^2) information here!)
        if (!large_scale){
    	    UPDATE_CURRENT_DIST_MATRIX(min_dim, -1);
	    }

        // check, if we can stop
        if (auto_feature){
            // the loss function that determines the selection is the FIRST one, so 
            // we simply use the first entries of selected_errors        
            if (optimal_performance(selected_errors, i, -1)){
                break;
            }
        }
	}
	
    // the final selected features
	DEBUG_PRINT("\nSelected features: ");
	for (i=0;i<dim;i++){
	    if (selected_dimensions[i]==1){
        	DEBUG_PRINT("%i ",i);
    	}	    
    }
    
	DEBUG_PRINT("\n");
    STOP_TIMER(1);
    RECORD_TIMER_SUM(1);
    
    DEBUG_PRINT("------------------------------------------------------------------------------------\n");            
    DEBUG_PRINT("Total time: \t\t\t\t\t\t\t\t%2.10f\n", (double)TIMER_SUM(1)/1000000);
    DEBUG_PRINT("------------------------------------------------------------------------------------\n");            
    DEBUG_PRINT("\tcompute_distance_matrix: \t\t\t\t\t%2.10f\n", (double)TIMER_SUM(2)/1000000);  
    DEBUG_PRINT("\tget_smallest_validation_error: \t\t\t\t\t%2.10f\n", (double)TIMER_SUM(3)/1000000);
    DEBUG_PRINT("\tcompute_predictions: \t\t\t\t\t\t%2.10f\n", (double)TIMER_SUM(5)/1000000);
    DEBUG_PRINT("\tcompute_losses: \t\t\t\t\t\t%2.10f\n", (double)TIMER_SUM(6)/1000000);               
    DEBUG_PRINT("\tupdate_current_dist_matrix: \t\t\t\t\t%2.10f\n", (double)TIMER_SUM(4)/1000000);
    DEBUG_PRINT("\tOverhead: \t\t\t\t\t\t\t%2.10f\n", (double)(TIMER_SUM(1)-TIMER_SUM(2)-TIMER_SUM(3)-TIMER_SUM(4)-TIMER_SUM(5)-TIMER_SUM(6))/1000000);
    DEBUG_PRINT("------------------------------------------------------------------------------------\n");

    // free memory and devices 
	FREE_MEMORY(large_scale);
    FREE_OPENCL_DEVICES();   
 	
	// free memory on host system
	free(val_errors);
    free(best_current_k_model_parameters);

	// Note: We do not free the selected_dimensions, since this information is used afterwards
	// DO NOT REMOVE: free(selected_dimensions);
	DEBUG_PRINT("\n");
    return num_final_features;
}

/* ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------------- 
 * Train: Forward Selection (MEDIUM)
 * -------------------------------------------------------------------------------- 
*/
int forward_selection(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, FLOAT_TYPE *Ytrain, int nYtrain, 
                      int *kvals, int num_kvals, int num_features, int kfold, int large_scale, 
                      int *loss_functions, int num_loss_functions, int auto_feature, int model_type) {
	DEBUG_PRINT("------------------------------------------\n");
	DEBUG_PRINT("Forward Feature Selection (LARGE_SCALE=%i, DEBUG=%i, USE_GPU=%i) \n", large_scale, DEBUG, USE_GPU);
	DEBUG_PRINT("------------------------------------------\n");

    // large-scale is not implemented yet ...
    if (large_scale){
        printf("Not yet implemented. Exiting ...\n");
        exit(0);
    }

    // counters
	int i,j,l;
    // set all parameters
	dim = dXtrain;
	training_patterns = Xtrain;	
	num_training_patterns = nXtrain;
	training_labels = Ytrain;
	num_training_labels = nYtrain;
	num_desired_features = num_features;
	// the final number of features (is the same as num_desired_features at the end if auto search is deactivated)
    num_final_features = 0;
	DEBUG_PRINT("Number of desired features=%i\n",num_desired_features);
    // automatic search for optimal feature dimension
    if (auto_feature){
    	DEBUG_PRINT("Automatic search for optimal feature dimension active!\n");
        num_desired_features = dim;
    }
    k_fold = kfold;
    DEBUG_PRINT("Number of folds=%i\n",k_fold);
	k_vals = kvals;
	num_k_vals = num_kvals;
	k_max = get_max_value(k_vals, num_k_vals);
    DEBUG_PRINT("k_max=%i\n",k_max);
    best_current_k_model_parameters = (int*)malloc(dim*sizeof(int));

    // init the OpenCl device (or do nothing in case USE_GPU==0)
    INIT_OPENCL_DEVICES();

    START_TIMER(1);
    // initialize all buffers and arrays
    INIT_MEMORY(large_scale);
    
	// initialize all dimensions that are selected (1=select, 0=not selected)
	selected_dimensions = (int*)malloc(dim*sizeof(int));
	for (i=0;i<dim;i++){selected_dimensions[i] = 0;}
    // initialize the ordering array (which features are selected first)
	selected_ordering = (int*)malloc(dim*sizeof(int));
	for (i=0;i<dim;i++){selected_ordering[i] = 0;}
    // initialize the error array (which contains, for each loss function specified,
    // the corresponding cross-validation error for each step)
	selected_errors = (FLOAT_TYPE*)malloc(num_loss_functions*dim*sizeof(FLOAT_TYPE));
	for (i=0;i<num_loss_functions*dim;i++){selected_errors[i] = 0.0;}
    if (!large_scale){
    	EMPTY_DISTANCE_MATRIX();
		DEBUG_PRINT("Empty distance matrix initialized ...\n\n");
	}

    // array containing the intermediate validation errors for each tested dimension
    FLOAT_TYPE *val_errors = (FLOAT_TYPE*)malloc(num_loss_functions*dim*sizeof(FLOAT_TYPE));

	// Perform iterative feature selection: In each round, we identify 
    // (and remove) a single dimension that does carry the least useful information
	for (i=0; i<num_desired_features; i++){
	    DEBUG_PRINT("Processing iteration %i (of %i)\n",i,num_desired_features);
        num_final_features++;
        FLOAT_TYPE min_val_error = MAX_FLOAT_TYPE;
        int min_dim = -1;
        
        // compute all validation errors 
        GET_SMALLEST_VALIDATION_ERRORS(selected_dimensions, val_errors, +1, large_scale, loss_functions, num_loss_functions, model_type);
        
	    // now get the smallest val error
	    for (j=0; j<dim; j++){
	        // if the dimension has not been added yet
	        if (selected_dimensions[j] == 0){
	            // add dimension
	            DEBUG_PRINT(" -  Dimension %i has val_error=%f\n",j,val_errors[j]);
	            // we use the first loss function for selecting the best dimension	            
	            if (val_errors[j] < min_val_error){
	                min_val_error = val_errors[j];
                    min_dim = j;
	            }	        
            }
        }

        // the last dimension that is added -> store corresponding optimal model parameter
        best_k_model_parameter = best_current_k_model_parameters[min_dim];

	    // add dimension min_dim 
	    DEBUG_PRINT(" -> Adding dimension %i!\n\n", min_dim);
	    selected_dimensions[min_dim] = 1;
	    selected_ordering[i] = min_dim;

	    // save all losses
	    for (l=0;l<num_loss_functions;l++){
    	    selected_errors[l*dim + i] = val_errors[l*dim + min_dim];
	    }	    
  
	    // update global distance matrix (we only need to change O(1 n^2) information here!)
        if (!large_scale){
    	    UPDATE_CURRENT_DIST_MATRIX(min_dim, +1);
	    }
	    
        // check, if we can stop
        if (auto_feature){
            // the loss function that determines the selection is the FIRST one, so 
            // we simply use the first entries of selected_errors
            if (optimal_performance(selected_errors, i, +1)){
                break;
            }
        }
	}

    // the final selected features	
	DEBUG_PRINT("Selected features: ");
	for (i=0;i<dim;i++){
	    if (selected_dimensions[i]==1){
        	DEBUG_PRINT("%i ",i);
    	}	    
    }
    
	DEBUG_PRINT("\n");
    STOP_TIMER(1);
    RECORD_TIMER_SUM(1);
    
    DEBUG_PRINT("------------------------------------------------------------------------------------\n");            
    DEBUG_PRINT("Total time: \t\t\t\t\t\t\t\t%2.10f\n", (double)TIMER_SUM(1)/1000000);
    DEBUG_PRINT("------------------------------------------------------------------------------------\n");            
    DEBUG_PRINT("\tcompute_distance_matrix: \t\t\t\t\t%2.10f\n", (double)TIMER_SUM(2)/1000000);  
    DEBUG_PRINT("\tget_smallest_validation_error: \t\t\t\t\t%2.10f\n", (double)TIMER_SUM(3)/1000000);
    DEBUG_PRINT("\tcompute_predictions: \t\t\t\t\t\t%2.10f\n", (double)TIMER_SUM(5)/1000000);
    DEBUG_PRINT("\tcompute_losses: \t\t\t\t\t\t%2.10f\n", (double)TIMER_SUM(6)/1000000);                
    DEBUG_PRINT("\tupdate_current_dist_matrix: \t\t\t\t\t%2.10f\n", (double)TIMER_SUM(4)/1000000);
    DEBUG_PRINT("\tOverhead: \t\t\t\t\t\t\t%2.10f\n", (double)(TIMER_SUM(1)-TIMER_SUM(2)-TIMER_SUM(3)-TIMER_SUM(4)-TIMER_SUM(5)-TIMER_SUM(6))/1000000);  
    DEBUG_PRINT("------------------------------------------------------------------------------------\n");            

    // free memory and devices    
	FREE_MEMORY(large_scale);    
    FREE_OPENCL_DEVICES();

	// free memory on host system
	free(val_errors);	
    free(best_current_k_model_parameters);

	// Note: We do not free the selected_dimensions, since this information is used afterwards
	// DO NOT REMOVE: free(selected_dimensions);
	DEBUG_PRINT("\n");
    return num_final_features;
}


/* -------------------------------------------------------------------------------- 
 * Get selected dimensions
 * -------------------------------------------------------------------------------- 
*/
void get_selected_dimensions(int *selected_dimensions_extern, int num_selected_dimensions_extern){
    int i;
    for (i=0;i<num_selected_dimensions_extern;i++){
        selected_dimensions_extern[i] = selected_dimensions[i];
    }
}

/* -------------------------------------------------------------------------------- 
 * Get selected dimensions
 * -------------------------------------------------------------------------------- 
*/
void get_selected_ordering(int *selected_ordering_extern, int num_selected_ordering_extern){
    int i;
    for (i=0;i<num_selected_ordering_extern;i++){
        selected_ordering_extern[i] = selected_ordering[i];
    }
}

/* -------------------------------------------------------------------------------- 
 * Get selected dimensions
 * -------------------------------------------------------------------------------- 
*/
void get_selected_errors(FLOAT_TYPE *selected_errors_extern, int num_selected_errors_extern){
    int i;
    for (i=0;i<num_selected_errors_extern;i++){
        selected_errors_extern[i] = selected_errors[i];
    }
}

int get_final_optimal_model_k(){
    return best_k_model_parameter;
}

/* -------------------------------------------------------------------------------- 
 * Computes all loss values
 * -------------------------------------------------------------------------------- 
*/
FLOAT_TYPE get_loss_all(FLOAT_TYPE *labels, int num_labels, FLOAT_TYPE *predictions, int num_predictions, int loss_function){
    return get_loss(labels, num_labels, predictions, num_predictions, loss_function);
}

/* -------------------------------------------------------------------------------- 
 * Used for automatic feature selection ("when to stop")
 * -------------------------------------------------------------------------------- 
*/
int optimal_performance(FLOAT_TYPE *errors, int iter, int mult){
    // forward selection
    if (mult>0){
        if (iter>0){
            if (errors[iter-1] < errors[iter]){
                return 1;
            }
        }
    }
    // backward selection
    if (mult<0){
        printf("iter=%i\n",iter);
        if (iter>0 && iter < dim-1){
            if (errors[iter+1] < errors[iter]){
                return 1;
            }
        }
    }
    return 0;
}

