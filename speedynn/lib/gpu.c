/* 
 * gpu.c
 */
#include "gpu.h"

/* -------------------------------------------------------------------------------- 
 * Initializes all devices at the beginning of the  querying process.
 * -------------------------------------------------------------------------------- 
*/
void init_opencl_devices(void){
    DEBUG_PRINT("Initializing GPU ...\n");
    init_opencl();
    DEBUG_PRINT("Compiling kernels ...\n");
    // define constants for kernels
    char constants[1000];
    sprintf (constants, "#define DIM %d\n#define WORKGROUP_SIZE %i\n#define WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR %i\n#define USE_LOCAL_MEM %i\n#define K_MAX %i\n#define NUM_K_VALS %i\n\0", \
                    dim, WORKGROUP_SIZE, WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR, USE_LOCAL_MEM, k_max, num_k_vals);

    // kernel KERNEL_NAME_COMPUTE_DISTANCE_MATRIX
    char* kernelSource; 
    unsigned long size;
    readfile(KERNEL_NAME_COMPUTE_DISTANCE_MATRIX, &kernelSource, &size);
    char outbuf[20000] = "";
    strcat(outbuf, constants);
    strncat(outbuf, kernelSource, size);
    strcat(outbuf, "\0");
    kernel_compute_distance_matrix = make_kernel(outbuf, "do_compute_distance_matrix");
    free(kernelSource);

    // kernel KERNEL_NAME_COMPUTE_DISTANCE_MATRIX_SELECTED
    readfile(KERNEL_NAME_COMPUTE_DISTANCE_MATRIX_SELECTED, &kernelSource, &size);
    strcat(outbuf, constants);
    strncat(outbuf, kernelSource, size);
    strcat(outbuf, "\0");
    kernel_compute_distance_matrix_selected = make_kernel(outbuf, "do_compute_distance_matrix_selected");
    free(kernelSource);

    // kernel KERNEL_UPDATE_CURRENT_DISTANCE_MATRIX
    readfile(KERNEL_NAME_UPDATE_CURRENT_DISTANCE_MATRIX, &kernelSource, &size);
    strcat(outbuf, constants);
    strncat(outbuf, kernelSource, size);
    strcat(outbuf, "\0");
    kernel_update_current_distance_matrix = make_kernel(outbuf, "do_update_current_distance_matrix");
    free(kernelSource);

    // kernel KERNEL_NAME_GET_SMALLEST_VALIDATION_ERROR
    readfile(KERNEL_NAME_GET_SMALLEST_VALIDATION_ERROR, &kernelSource, &size);
    strcat(outbuf, constants);
    strncat(outbuf, kernelSource, size);
    strcat(outbuf, "\0");
    kernel_get_smallest_validation_error = make_kernel(outbuf, "do_get_smallest_validation_error");
    kernel_compute_dim_removed_matrix = make_kernel(outbuf, "compute_dim_removed_matrix");    
    free(kernelSource);  

    // kernel KERNEL_NAME_COMPUTE_PREDICTIONS
    readfile(KERNEL_NAME_COMPUTE_PREDICTIONS, &kernelSource, &size);
    strcat(outbuf, constants);
    strncat(outbuf, kernelSource, size);
    strcat(outbuf, "\0");
    kernel_compute_predictions_regression = make_kernel(outbuf, "do_compute_predictions_regression");
    kernel_compute_predictions_classification = make_kernel(outbuf, "do_compute_predictions_classification");    
    free(kernelSource);  
    DEBUG_PRINT("GPU initialized successfully!\n");
}

/* -------------------------------------------------------------------------------- 
 * After having performed all queries: Free memory etc.
 * -------------------------------------------------------------------------------- 
*/
void free_opencl_devices(void){
    // free kernels
    clReleaseKernel(kernel_compute_distance_matrix);        
    clReleaseKernel(kernel_compute_distance_matrix_selected);            
    clReleaseKernel(kernel_update_current_distance_matrix);
    clReleaseKernel(kernel_get_smallest_validation_error);
    clReleaseKernel(kernel_compute_dim_removed_matrix);    
    clReleaseKernel(kernel_compute_predictions_regression);
    clReleaseKernel(kernel_compute_predictions_classification); 
   
    // free opencl
    free_opencl();
    //DEBUG_PRINT("GPU resources released ...\n");        
}

void init_memory_gpu(int large_scale){
    cl_int err;
    if (!large_scale) {
        // array that contains, for each training pattern, the associated cross validation errors for each k value
        intermediate_predictions = (FLOAT_TYPE*)malloc(num_training_patterns*num_k_vals*sizeof(FLOAT_TYPE));

        // READ_ONLY: training labels
        device_training_labels = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, \
                                num_training_patterns*sizeof(FLOAT_TYPE), training_labels, &err);  
        check_cl_error(err, __FILE__, __LINE__);

        // READ_ONLY: training patterns (by transposing the training patterns, we can achieve 
        // cached access on the GPU, see the kernel code)
        int transpose=1;
        if (transpose){
            FLOAT_TYPE *training_patterns_transposed = (FLOAT_TYPE*)malloc(num_training_patterns*dim*sizeof(FLOAT_TYPE));
            int i,j;
            for (j=0; j<dim; j++){
                for (i=0; i<num_training_patterns; i++){
                    training_patterns_transposed[j*num_training_patterns + i] =  training_patterns[i*dim+j];
                }
            }
            device_training_patterns = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, \
                                    num_training_patterns*dim*sizeof(FLOAT_TYPE), training_patterns_transposed, &err);
            check_cl_error(err, __FILE__, __LINE__);
        } else {
            device_training_patterns = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, \
                                    num_training_patterns*dim*sizeof(FLOAT_TYPE), training_patterns, &err);
            check_cl_error(err, __FILE__, __LINE__);
        }

        // READ_ONLY: k values
        device_k_vals = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, num_k_vals*sizeof(int), k_vals, &err);
        check_cl_error(err, __FILE__, __LINE__);

        // READ_WRITE: current distance matrix
        device_current_dist_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, \
                                num_training_patterns*num_training_patterns*sizeof(FLOAT_TYPE), NULL, &err);
        check_cl_error(err, __FILE__, __LINE__);                                

        // READ_WRITE: predictions
        device_intermediate_predictions = clCreateBuffer(context, CL_MEM_READ_WRITE, \
                                num_training_patterns*num_k_vals*sizeof(FLOAT_TYPE), NULL, &err);        
        check_cl_error(err, __FILE__, __LINE__);

        // READ_WRITE: buffers for intermediate distances and indices        
        // This buffer can get too large (e.g., if a large k and many dimensions are just). In this case, 
        // just traverse the loop below (over the dimensions) in batches of fixed size (e.g., 10 or 20).
        if (num_training_patterns*k_max*dim*sizeof(int) / 1000000 >= 1000){
            // Is OK for, e.g., n=10000, k=20, dim=500, float -> 400MB        
            printf("WARNING: More than 1 GB of data used for intermediate results!");
        } 
        device_k_max_nearest_neighbors_indices_global = clCreateBuffer(context, CL_MEM_READ_WRITE, \
                                num_training_patterns*k_max*dim*sizeof(int), NULL, &err);
        check_cl_error(err, __FILE__, __LINE__);          
        

                                                      
    }
    DEBUG_PRINT("GPU memory allocated successfully!\n");
}

void free_memory_gpu(int large_scale){
    // free memory on GPU
    clReleaseMemObject(device_training_patterns);
    clReleaseMemObject(device_training_labels);      
    clReleaseMemObject(device_k_vals);      
    clReleaseMemObject(device_current_dist_matrix);
    clReleaseMemObject(device_intermediate_predictions);    
    clReleaseMemObject(device_k_max_nearest_neighbors_indices_global);
    //DEBUG_PRINT("GPU memory released!\n");
    
    // free memory on host system
    free(intermediate_predictions);
}

void get_smallest_validation_errors_gpu(int *selected_dimensions, FLOAT_TYPE *val_errors, int mult, int large_scale, int *loss_functions, int num_loss_functions, int model_type){
    // counters
    int i,j,m,l;
    cl_event event;
    cl_int err;

    int check_dim;
    if (mult>0){ 
        // forward selection
        check_dim=0;
    }else{ 
        // backward_selection
        check_dim=1;
    }

    // this array contains the cross validation errors (sum over the folds) for each k value
    FLOAT_TYPE *k_errors = (FLOAT_TYPE*)calloc(num_loss_functions*num_k_vals,sizeof(FLOAT_TYPE));		  

    START_TIMER(3);
    int current_dimension=0;
    int *active_dims = (int*)calloc(dim, sizeof(int));
    int num_active_dims = 0;
    for (current_dimension=0; current_dimension<dim; current_dimension++){
        // if the dimension has not been removed yet
        if (selected_dimensions[current_dimension] == check_dim){        
            active_dims[num_active_dims] = current_dimension;
            num_active_dims++;
        }
    }
    device_active_dimensions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              num_active_dims * sizeof(int), active_dims, &err);

    // compute all validation errors
    size_t globalSize[2]={WORKGROUP_SIZE*((int)num_training_patterns/WORKGROUP_SIZE) + WORKGROUP_SIZE, num_active_dims};
    size_t localSize[2]={WORKGROUP_SIZE,1};

    // number of elements per fold (except last one, which can be larger since we use the rounded (int) value)
    int num_per_fold = (int) (num_training_patterns / k_fold);

    // set kernel parameters: kernel_get_smallest_validation_error
    err =  clSetKernelArg(kernel_get_smallest_validation_error, 0, sizeof(cl_mem), &device_training_patterns);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_get_smallest_validation_error, 1, sizeof(cl_mem), &device_training_labels);
    check_cl_error(err, __FILE__, __LINE__);    
    err = clSetKernelArg(kernel_get_smallest_validation_error, 2, sizeof(cl_mem), &device_current_dist_matrix);
    check_cl_error(err, __FILE__, __LINE__);
    err = clSetKernelArg(kernel_get_smallest_validation_error, 3, sizeof(cl_mem), &device_k_vals);
    check_cl_error(err, __FILE__, __LINE__);            
    err = clSetKernelArg(kernel_get_smallest_validation_error, 4, sizeof(cl_mem), &device_k_max_nearest_neighbors_indices_global);
    check_cl_error(err, __FILE__, __LINE__);     
    err =  clSetKernelArg(kernel_get_smallest_validation_error, 5, sizeof(int), &num_training_patterns);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_get_smallest_validation_error, 6, sizeof(int), &num_per_fold);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_get_smallest_validation_error, 7, sizeof(int), &current_dimension);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_get_smallest_validation_error, 8, sizeof(int), &mult);
    check_cl_error(err, __FILE__, __LINE__);
    err = clSetKernelArg(kernel_get_smallest_validation_error, 9, sizeof(cl_mem), &device_active_dimensions);
    check_cl_error(err, __FILE__, __LINE__);     
    
    err = clEnqueueNDRangeKernel(commandQueue, kernel_get_smallest_validation_error, 2, NULL, \
                                globalSize, localSize, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
    err = clWaitForEvents(1, &event);
    check_cl_error(err, __FILE__, __LINE__);
    err = clReleaseEvent(event);
    check_cl_error(err, __FILE__, __LINE__);

    clReleaseMemObject(device_active_dimensions);
    free(active_dims);
    STOP_TIMER(3);
    RECORD_TIMER_SUM(3);
    
    // We have found the nearest neighbor indices so far. Now, compute the predictions (this is different for
    // regression and classification scenarios...)
    for (current_dimension=0; current_dimension<dim; current_dimension++){
        // if the dimension has not been removed yet
        if (selected_dimensions[current_dimension] == check_dim){
            // second kernel, get cross validation errors
            size_t globalSize[]={WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR*((int)num_training_patterns/WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR) + WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR};
            size_t localSize[]={WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR};    
            if (model_type==REGRESSION_MODEL){
                START_TIMER(5);
                // set kernel parameters: kernel_compute_predictions_regression
                err =  clSetKernelArg(kernel_compute_predictions_regression, 0, sizeof(cl_mem), &device_training_labels);
                check_cl_error(err, __FILE__, __LINE__);    
                err = clSetKernelArg(kernel_compute_predictions_regression, 1, sizeof(cl_mem), &device_intermediate_predictions);
                check_cl_error(err, __FILE__, __LINE__);        
                err = clSetKernelArg(kernel_compute_predictions_regression, 2, sizeof(cl_mem), &device_k_vals);
                check_cl_error(err, __FILE__, __LINE__);            
                err = clSetKernelArg(kernel_compute_predictions_regression, 3, sizeof(cl_mem), &device_k_max_nearest_neighbors_indices_global);
                check_cl_error(err, __FILE__, __LINE__);
                err =  clSetKernelArg(kernel_compute_predictions_regression, 4, sizeof(int), &num_training_patterns);
                check_cl_error(err, __FILE__, __LINE__);
                err =  clSetKernelArg(kernel_compute_predictions_regression, 5, sizeof(int), &current_dimension);
                check_cl_error(err, __FILE__, __LINE__);
                
                // execute kernel
                err = clEnqueueNDRangeKernel(commandQueue, kernel_compute_predictions_regression, 1, NULL, \
                                            globalSize, localSize, 0, NULL, &event);
                check_cl_error(err, __FILE__, __LINE__);
                err = clWaitForEvents(1, &event);
                check_cl_error(err, __FILE__, __LINE__);
                err = clReleaseEvent(event);
                check_cl_error(err, __FILE__, __LINE__);
                STOP_TIMER(5);
                RECORD_TIMER_SUM(5);
                
                START_TIMER(6);
                // compute predictions
                err = clEnqueueReadBuffer(commandQueue, device_intermediate_predictions, CL_TRUE, 0, \
                                        num_training_patterns*num_k_vals*sizeof(FLOAT_TYPE), intermediate_predictions, 0, NULL, &event);
                check_cl_error(err, __FILE__, __LINE__);
                err = clWaitForEvents(1, &event);
                check_cl_error(err, __FILE__, __LINE__);
                err = clReleaseEvent(event);
                check_cl_error(err, __FILE__, __LINE__);
                // compute the losses (might be a bit more time-consuming)
                // TODO: do this on GPU for many loss functions ...
                for (l=0;l<num_loss_functions;l++){
                    for (m=0; m<num_k_vals; m++){
                        k_errors[l*num_k_vals + m] = get_loss(training_labels, num_training_patterns, intermediate_predictions+m*num_training_patterns, num_training_patterns, loss_functions[l]);
                    }
                }
                STOP_TIMER(6);
                RECORD_TIMER_SUM(6);
            } else if (model_type==CLASSIFICATION_MODEL) {
                printf("model type not implemented ...\n");
                exit(0);                
            } else {
                printf("model type not implemented ...\n");
                exit(0);
            }
                        
            // compute the smallest (==best) validation error (depending
            // on the first loss function given)
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
        }
    }
    // free memory on host system
    free(k_errors);
}

void update_current_dist_matrix_gpu(int min_dim, int mult){
    START_TIMER(4);
    int i,j;
    cl_event event;
    cl_int err;
    // set kernel parameters: kernel_compute_distance_matrix
    err =  clSetKernelArg(kernel_update_current_distance_matrix, 0, sizeof(cl_mem), &device_training_patterns);
    check_cl_error(err, __FILE__, __LINE__);
    err = clSetKernelArg(kernel_update_current_distance_matrix, 1, sizeof(cl_mem), &device_current_dist_matrix);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_update_current_distance_matrix, 2, sizeof(int), &num_training_patterns);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_update_current_distance_matrix, 3, sizeof(int), &min_dim);
    check_cl_error(err, __FILE__, __LINE__);
    err =  clSetKernelArg(kernel_update_current_distance_matrix, 4, sizeof(int), &mult);
    check_cl_error(err, __FILE__, __LINE__);

    // execute kernel
    size_t globalSize[]={WORKGROUP_SIZE*((int)num_training_patterns/WORKGROUP_SIZE) + WORKGROUP_SIZE};
    size_t localSize[]={WORKGROUP_SIZE};
    err = clEnqueueNDRangeKernel(commandQueue, kernel_update_current_distance_matrix, 1, NULL, \
                                globalSize, localSize, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
    err = clWaitForEvents(1, &event);
    check_cl_error(err, __FILE__, __LINE__);
    err = clReleaseEvent(event);
    check_cl_error(err, __FILE__, __LINE__);
    STOP_TIMER(4);    
    RECORD_TIMER_SUM(4);
}

void empty_distance_matrix_gpu(){
    FLOAT_TYPE* current_dist_m = (FLOAT_TYPE*)calloc(num_training_patterns*num_training_patterns, sizeof(FLOAT_TYPE));
    cl_int err;
    clReleaseMemObject(device_current_dist_matrix);
    device_current_dist_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, num_training_patterns*num_training_patterns*sizeof(FLOAT_TYPE), current_dist_m, &err);
    check_cl_error(err, __FILE__, __LINE__);
    free(current_dist_m);
}

void compute_distance_matrix_gpu(int *selected_dimensions, int selected_only){
    // We might gain a large speed-up here in case we are only updating the old (given) information; otherwise,
    // we have to compute the matrix based on data in global memory, which has to be loaded each time!
    START_TIMER(2);
    
    cl_event event;   
    cl_int err;
    
    if (!selected_only){
        // set kernel parameters: kernel_compute_distance_matrix
        err =  clSetKernelArg(kernel_compute_distance_matrix, 0, sizeof(cl_mem), &device_training_patterns);
        check_cl_error(err, __FILE__, __LINE__);
        err = clSetKernelArg(kernel_compute_distance_matrix, 1, sizeof(cl_mem), &device_current_dist_matrix);
        check_cl_error(err, __FILE__, __LINE__);
        err =  clSetKernelArg(kernel_compute_distance_matrix, 2, sizeof(int), &num_training_patterns);
        check_cl_error(err, __FILE__, __LINE__);

        // execute kernel
        size_t globalSize[]={WORKGROUP_SIZE*((int)num_training_patterns/WORKGROUP_SIZE) + WORKGROUP_SIZE};
        size_t localSize[]={WORKGROUP_SIZE};        
        err = clEnqueueNDRangeKernel(commandQueue, kernel_compute_distance_matrix, 1, NULL, \
                                    globalSize, localSize, 0, NULL, &event);        
        check_cl_error(err, __FILE__, __LINE__);
        err = clWaitForEvents(1, &event);
        check_cl_error(err, __FILE__, __LINE__);        
        clReleaseEvent(event);
    } else {
        // copy selected dimensions to GPU (if copying the whole array is slow, then copy the indices only)
        cl_mem device_selected_dimensions = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, \
                            dim*sizeof(int), selected_dimensions, &err);
        check_cl_error(err, __FILE__, __LINE__);
        // set kernel parameters: kernel_compute_distance_matrix_selected
        err =  clSetKernelArg(kernel_compute_distance_matrix_selected, 0, sizeof(cl_mem), &device_training_patterns);
        check_cl_error(err, __FILE__, __LINE__);
        err = clSetKernelArg(kernel_compute_distance_matrix_selected, 1, sizeof(cl_mem), &device_current_dist_matrix);
        check_cl_error(err, __FILE__, __LINE__);
        err = clSetKernelArg(kernel_compute_distance_matrix_selected, 2, sizeof(cl_mem), &device_selected_dimensions);
        check_cl_error(err, __FILE__, __LINE__);
        err =  clSetKernelArg(kernel_compute_distance_matrix_selected, 3, sizeof(int), &num_training_patterns);
        check_cl_error(err, __FILE__, __LINE__);

        // execute kernel
        size_t globalSize[]={WORKGROUP_SIZE*((int)num_training_patterns/WORKGROUP_SIZE) + WORKGROUP_SIZE};
        size_t localSize[]={WORKGROUP_SIZE};
        err = clEnqueueNDRangeKernel(commandQueue, kernel_compute_distance_matrix_selected, 1, NULL, \
                                    globalSize, localSize, 0, NULL, &event);
        check_cl_error(err, __FILE__, __LINE__);
        err = clWaitForEvents(1, &event);
        check_cl_error(err, __FILE__, __LINE__);
        clReleaseEvent(event);
        
        clReleaseMemObject(device_selected_dimensions);
            
    }        
    STOP_TIMER(2);
    RECORD_TIMER_SUM(2);
}


void readfile(char * filename, char ** text, unsigned long * size)
{
    FILE *fp;
    char ch;  

    fp = fopen(filename, "r");
    if (fp == NULL) {
       printf("I couldn't open the file for reading.\n");
       exit(0);
    }
    
    if (fseek(fp, 0, SEEK_END) == 0)
    {
        *size = ftell(fp);
        // go back to start of file
        fseek(fp, 0, SEEK_SET);
    }
    *text = (char*)malloc(sizeof(char)*(*size));
    int i=0;
    while( ( ch = fgetc(fp) ) != EOF ){
        (*text)[i] = ch;
        i++;
    }
    
    fclose(fp);
}

void check_cl_error(cl_int err, const char * file, int line) {
    if (err != CL_SUCCESS) {
        printf("Error with errorcode: %d in file %s in line %d \n", err, file, line);
        exit(1);
    }
}

void print_info(){
    cl_uint i;
    cl_int err;

    // get number of platforms
    cl_uint nplatforms;
    err = clGetPlatformIDs(0, NULL, &nplatforms);
    check_cl_error(err, __FILE__, __LINE__);

    // get platform ids
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nplatforms);
    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    check_cl_error(err, __FILE__, __LINE__);

    char name[128];
    char vendor[128];
    char version[128];

    fprintf(stdout, "Number of detected OpenCL platforms: %d\n", nplatforms);

    for (i = 0; i < nplatforms; i++) {
        err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 128, vendor, NULL);
        err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, name, NULL);
        err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 128, version, NULL);
        check_cl_error(err, __FILE__, __LINE__);
        fprintf(stdout, "\tPlatform %d: %s %s %s\n", i, vendor, name, version);
    }

    free(platforms);
}

void print_build_log(cl_program program, cl_device_id device) {
    cl_int err;
    char* build_log;
    size_t build_log_size;
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
    check_cl_error(err, __FILE__, __LINE__);
    build_log = (char*) malloc(build_log_size);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
    check_cl_error(err, __FILE__, __LINE__);
    free(build_log);
}

void init_opencl() {
    cl_int err;
    cl_uint nplatforms;
    // FIXME: this init phase is slow on some systems (e.g., access via ssh). driver issue?
    // For runtime experiments, this is not a huge problem (since it only has to be done once, 
    // e.g., while parsing the input data or at startup).
    err = clGetPlatformIDs(0, NULL, &nplatforms);
    check_cl_error(err, __FILE__, __LINE__);
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nplatforms);
    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    check_cl_error(err, __FILE__, __LINE__);
    // save device
    err = clGetDeviceIDs(platforms[GPU_PLATFORM_NUMBER], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    check_cl_error(err, __FILE__, __LINE__);
    // generate context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl_error(err, __FILE__, __LINE__);
    // generate command queue
    commandQueue = clCreateCommandQueue(context, device, 0, &err);
    check_cl_error(err, __FILE__, __LINE__);
    cl_ulong size;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
    //printf("Local Memory Size=%i\n",size);    
}

void free_opencl() {
    cl_int err;
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
}

cl_kernel make_kernel(const char * kernelSource, const char * kernelName)	{
    cl_int err;
    cl_kernel kernel;
    // length of kernel code
    size_t sourceLength = (size_t) strlen(kernelSource); 
    cl_program program;
    
    // generate program
    program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceLength, &err);
    check_cl_error(err, __FILE__, __LINE__);

    // build program for device
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
        print_build_log(program, device);
    else{
        print_build_log(program, device);
    }

    kernel = clCreateKernel(program, kernelName, &err);
    check_cl_error(err, __FILE__, __LINE__);
    clReleaseProgram(program);
    return kernel;
}



