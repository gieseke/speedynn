#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38

__kernel void do_compute_distance_matrix_selected(
					__global FLOAT_TYPE* training_patterns,
					__global FLOAT_TYPE* current_dist_matrix,
					__global int* selected_dimensions,
					int num_training_patterns
					)
{       
    // get global thread id
    int tid=get_global_id(0);
    if(tid>=num_training_patterns){return;}
    
    // counters
    int j,k;

    FLOAT_TYPE tmp;
    FLOAT_TYPE dist_tmp;

    // generate private copy of current column (tid) pattern
    FLOAT_TYPE patt_current_row[DIM];
    for(k=DIM;k--;){
        patt_current_row[k] = training_patterns[k*num_training_patterns + tid];
    }

    // compute col 'tid' of kernel matrix
    for (j=0; j<num_training_patterns; j++){
        dist_tmp = 0.0;
        // compute distance
        for (k=DIM;k--;){
            // only if dimensions is "active" (this is the SAME for all threads invoked!)
            if (selected_dimensions[k] == 1){
                // access to patt_current_row is fast (private copy); access to 
                // training_patterns is fast (caching, all threads access the same location)
                tmp = training_patterns[k*num_training_patterns + tid] - training_patterns[k*num_training_patterns+j];
                dist_tmp += tmp*tmp;
            }
        }
        // coalesced access to distance matrix
        current_dist_matrix[j*num_training_patterns + tid] = dist_tmp;
    }
    
}
