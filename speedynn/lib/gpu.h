/* 
 * gpu.h
 */
#ifndef GPU
#define GPU

#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "loss_functions.h"
#include "extern.h"
#include "util.h"
#include "nearest_neighbors_feature_selection.h"
#include <CL/cl.h>


FLOAT_TYPE *intermediate_predictions;

// global buffers
cl_mem device_training_patterns;
cl_mem device_training_labels;
cl_mem device_current_dist_matrix;
cl_mem device_intermediate_predictions;
cl_mem device_k_vals;
cl_mem device_k_max_nearest_neighbors_indices_global;
cl_mem device_active_dimensions;

// kernels
cl_kernel kernel_compute_distance_matrix;
cl_kernel kernel_compute_distance_matrix_selected;
cl_kernel kernel_update_current_distance_matrix;
cl_kernel kernel_get_smallest_validation_error;
cl_kernel kernel_compute_dim_removed_matrix;
cl_kernel kernel_compute_predictions_regression;
cl_kernel kernel_compute_predictions_classification;

/* -------------------------------------------------------------------------------- 
 * Initializes all devices at the beginning of the  querying process.
 * -------------------------------------------------------------------------------- 
*/
void init_opencl_devices(void);

/* -------------------------------------------------------------------------------- 
 * Allocates memory for testing phase.
 * -------------------------------------------------------------------------------- 
*/
void init_memory_gpu(int large_scale);

/* -------------------------------------------------------------------------------- 
 * After having performed all queries: Free memory etc.
 * -------------------------------------------------------------------------------- 
*/
void free_opencl_devices(void);

void get_smallest_validation_errors_gpu(int *selected_dimensions, FLOAT_TYPE *val_errors, int mult, int large_scale, int *loss_functions, int num_loss_functions, int model_type);
void compute_distance_matrix_gpu(int *selected_dimensions, int selected_only);
void empty_distance_matrix_gpu(void);
void update_current_dist_matrix_gpu(int min_dim, int mult);
void init_distance_matrix_cpu(void);
void free_memory_gpu(int large_scale);

// global opencl variables
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue commandQueue;

void readfile(char * filename, char ** text, unsigned long * size);
void check_cl_error(cl_int err, const char * file, int line); 
void print_info(void);
void print_build_log(cl_program program, cl_device_id device);
void init_opencl(void); 
void free_opencl(void); 
cl_kernel make_kernel(const char * kernelSource, const char * kernelName);

#endif
