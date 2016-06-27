#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38

__kernel void do_compute_predictions_regression(
					__global float* training_labels,
					__global float* validation_predictions,
					__global int* k_vals,				
					__global int* k_max_nearest_neighbors_indices_global,							
					int num_training_patterns,
					int dim_removed
					)
{   
    // get global thread id
    int tid=get_global_id(0);
    // get local workgroup id
    int lid = get_local_id(0);
    if(tid>=num_training_patterns){return;}

    // get target value of current pattern
    FLOAT_TYPE target_val = training_labels[tid];

    // counters
    int i,j,k,idx;

    FLOAT_TYPE tmp;
    // for each k value 
    for (i=0; i<NUM_K_VALS; i++){
        // memory access (caching!)
        k=k_vals[i];

        // compute predictions for each k value based on the k nearest neighbors
        FLOAT_TYPE pred_val = 0.0;
        for (j=0; j<k; j++){
            // coalesced access: the k nearest neighbors are stored in the first k slots of k_max_nearest_neighbors_indices
            idx = k_max_nearest_neighbors_indices_global[dim_removed*K_MAX*num_training_patterns + tid + j*num_training_patterns];
            // access to read-only training labels (constant memory)
            pred_val += training_labels[idx];
        }
        pred_val /= k;

        // store prediction in buffer
        validation_predictions[i*num_training_patterns+tid] = pred_val;
    }
}


__kernel void do_compute_predictions_classification(
					__global float* training_labels,
					__global float* validation_predictions,
					__global int* k_vals,				
					__global int* k_max_nearest_neighbors_indices_global,							
					int num_training_patterns,
					int dim_removed
					)
{   
    // FIXME: NOT ADAPTED TO CLASSIFICATION YET!

    // get global thread id
    int tid=get_global_id(0);
    // get local workgroup id
    int lid = get_local_id(0);
    if(tid>=num_training_patterns){return;}

    // get target value of current pattern
    FLOAT_TYPE target_val = training_labels[tid];

    // counters
    int i,j,k,idx;

    FLOAT_TYPE tmp;
    // for each k value 
    for (i=0; i<NUM_K_VALS; i++){
        // constant memory access (caching!)
        k=k_vals[i];

        // compute predictions for each k value based on the k nearest neighbors
        FLOAT_TYPE pred_val = 0.0;
        for (j=0; j<k; j++){
            // coalesced access: the k nearest neighbors are stored in the first k slots of k_max_nearest_neighbors_indices
            idx = k_max_nearest_neighbors_indices_global[dim_removed*K_MAX*num_training_patterns + tid + j*num_training_patterns];
            // access to read-only training labels (constant memory)
            // FIXME: NOT ADAPTED TO CLASSIFICATION YET!
            pred_val += training_labels[idx];
        }
        // FIXME: NOT ADAPTED TO CLASSIFICATION YET!
        pred_val /= k;

        // store prediction in buffer
        validation_predictions[i*num_training_patterns+tid] = pred_val;
    }
}


