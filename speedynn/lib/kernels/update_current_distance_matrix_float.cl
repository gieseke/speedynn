#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38

__kernel void do_update_current_distance_matrix(
					__global float* training_patterns,
					__global float* current_dist_matrix,
					int num_training_patterns,
					int min_dim,
					int mult
					)
{   
    // get global thread id
    int tid=get_global_id(0);
    if(tid>=num_training_patterns){return;}

    // counter
    int j;
    
    // private copy
    FLOAT_TYPE patt_current_row_dim_removed = training_patterns[min_dim*num_training_patterns + tid];
    
    FLOAT_TYPE tmp;
    // compute col 'tid' of matrix
    // TODO: parallize here too!
    for (j=0; j<num_training_patterns; j++){
        // fast access to private variable, access to training_patterns is cached!
        tmp = patt_current_row_dim_removed - training_patterns[min_dim*num_training_patterns + j];
        // update the current distance matrix (coalesced access)
        current_dist_matrix[j*num_training_patterns + tid] = current_dist_matrix[j*num_training_patterns + tid] + mult*tmp*tmp;
    }
}
