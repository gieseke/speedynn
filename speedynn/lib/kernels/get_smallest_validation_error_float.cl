#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38

// LOCAL_MEM TOO SLOW (not tested!)
#if USE_LOCAL_MEM > 0
    #define INIT_TEMP_MEMORY; __local int k_max_nearest_neighbors_indices_local[K_MAX*WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR]; __local FLOAT_TYPE k_max_nearest_neighbors_distances_local[K_MAX*WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR]; __local int *k_max_nearest_neighbors_indices=k_max_nearest_neighbors_indices_local +lid*K_MAX; __local FLOAT_TYPE *k_max_nearest_neighbors_distances = k_max_nearest_neighbors_distances_local + lid*K_MAX;
#else
    #define INIT_TEMP_MEMORY; int k_max_nearest_neighbors_indices[K_MAX]; FLOAT_TYPE k_max_nearest_neighbors_distances[K_MAX];
#endif


__kernel void compute_dim_removed_matrix(
					__global float* training_patterns,
					__global float* dim_removed_matrix,	
					int num_training_patterns,					
					int dim_removed		
){
    // get global thread id
    int tid=get_global_id(0);
    // get local workgroup id
    // int lid = get_local_id(0);
    if(tid>=num_training_patterns){return;}

    // counter
    int i;    
    // this value is needed below, in each iteration -> store it!
    FLOAT_TYPE tid_dim_val = training_patterns[dim_removed*num_training_patterns+tid];
    //FLOAT_TYPE d;

    for (i=0;i<num_training_patterns;i++){
        dim_removed_matrix[i*num_training_patterns + tid] = (training_patterns[dim_removed*num_training_patterns+i] - tid_dim_val);
    }
}

__kernel void do_get_smallest_validation_error(
					__global float* training_patterns,
					__global float* training_labels,
					__global float* current_dist_matrix,					
					__global int* k_vals,		
					__global int* k_max_nearest_neighbors_indices_global,
					int num_training_patterns,
					int num_per_fold,
					int dim_removed, 
					int mult,
					__global int* active_dimensions
					)
{   
    // get global thread id
    int tid=get_global_id(0);
    int dim_removed2 = active_dimensions[get_global_id(1)];
    // get local workgroup id
    // int lid = get_local_id(0);
    if(tid>=num_training_patterns){return;}
    
    // compute fold of training pattern
    int fold_of_pattern = (int) tid / num_per_fold;

    // counters
    int i,j;

    // Using these array can slow down local arrays (indices and distances)
    INIT_TEMP_MEMORY;
    for (j=K_MAX; j--; ){k_max_nearest_neighbors_distances[j]=MAX_FLOAT_TYPE; k_max_nearest_neighbors_indices[j]=0;}
    
    FLOAT_TYPE tmp;
    int tmp_idx;

    // this value is needed below, in each iteration -> store it!
    FLOAT_TYPE tid_dim_val = training_patterns[dim_removed2*num_training_patterns+tid];
    FLOAT_TYPE d;

    for (i=0;i<num_training_patterns;i++){
        // We make use of caching here: consecutive threads (almost) access the same memory positions
        // in the global array training_patterns; this access is faster than a coalesced access (e.g., 
        // one can precompute these values in a matrix). Further, training_patterns is READ_ONLY.
        tmp = (training_patterns[dim_removed2*num_training_patterns+i] - tid_dim_val);
        // The alternative scheme (using the kernel above) is actually slower ...
        //FLOAT_TYPE tmp2 = dim_removed_matrix[i*num_training_patterns + tid];
        
        // coaclesced access for the distance matrix: we cannot make use of caching
        // accessing this matrix takes quite a long time...(e.g., 40% of the whole kernel time) 
        d = current_dist_matrix[i*num_training_patterns + tid] + mult*tmp*tmp;
        //d = d+tmp2;
        //d = d - tmp2;

        // ingore all values that are within the same fold
        if (i / num_per_fold != fold_of_pattern){
            // Insert dist/idx in private/local arrays: We cannot do these
            // steps here much better. Given all the distances, we need to 
            // update the k nearest neighbors for each of the test/validation
            // indices...
            j=K_MAX-1;
            if(k_max_nearest_neighbors_distances[j]>d){
                // the code below is not executed very often ...
                k_max_nearest_neighbors_distances[j]=d;
                k_max_nearest_neighbors_indices[j]=i;
                for(;j>0;j--) {
                    if(k_max_nearest_neighbors_distances[j]<k_max_nearest_neighbors_distances[j-1]){
	                    //swap dist
	                    tmp=k_max_nearest_neighbors_distances[j];
	                    k_max_nearest_neighbors_distances[j]=k_max_nearest_neighbors_distances[j-1];
	                    k_max_nearest_neighbors_distances[j-1]=tmp;
	                    //swap idx
	                    tmp_idx=k_max_nearest_neighbors_indices[j];
	                    k_max_nearest_neighbors_indices[j]=k_max_nearest_neighbors_indices[j-1];
	                    k_max_nearest_neighbors_indices[j-1]=tmp_idx;
                    } //else break; // using break; makes it slower ...
                }
            }
        }
    }

    // write distances and indices to global (temporary) buffer
    for (j=K_MAX;j--;){
        // coalesced access to global buffers
        k_max_nearest_neighbors_indices_global[dim_removed2*K_MAX*num_training_patterns + j*num_training_patterns + tid] = k_max_nearest_neighbors_indices[j];
    }        
 

}


