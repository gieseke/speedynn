/* 
 * util.c
 */
#include "util.h"

/* -------------------------------------------------------------------------------- 
 * Computes the distance between point a and b in R^dim
 * -------------------------------------------------------------------------------- 
*/
FLOAT_TYPE dist(FLOAT_TYPE *a, FLOAT_TYPE *b, int dim){
    int i;
    FLOAT_TYPE d = 0.0;
    for (i=dim;i--;){
        d += (a[i]-b[i])*(a[i]-b[i]);
    }
    return d;    
}

FLOAT_TYPE dist_dimension(FLOAT_TYPE *a, FLOAT_TYPE *b, int dim_selected){
    return (a[dim_selected]-b[dim_selected])*(a[dim_selected]-b[dim_selected]);    
}

int get_max_value(int *array, int size_array){
    int i;
    int max_val = -10000000000;
    for (i=0; i<size_array; i++){
        if (array[i] > max_val){
            max_val = array[i];
        }
    }
    return max_val;
}

/*
 * Helper method for computing the current time (w.r.t to an offset).
 */
long get_system_time_in_microseconds(void){
	struct timeval tempo;
	gettimeofday(&tempo, NULL);
	return tempo.tv_sec * 1000000 + tempo.tv_usec;	
}


void matrix_multiply(FLOAT_TYPE *A, FLOAT_TYPE *B, FLOAT_TYPE *C, int A_n_rows, int A_n_cols, int B_n_cols, FLOAT_TYPE multiplier){
    // A_n_cols == B_n_rows
    int i,j,k;
    for (i = 0; i < A_n_rows; i++){
        for (j = 0; j < B_n_cols; j++){
            C[i*B_n_cols + j] = 0.0;
            for (k = 0; k < A_n_cols; k++){
                C[i*B_n_cols + j] += A[i*A_n_cols + k]*B[k*B_n_cols + j];
            }
            C[i*B_n_cols + j] *= multiplier;
        }
    }    
}
 
 
void compute_transpose(FLOAT_TYPE *A, FLOAT_TYPE *A_transpose, int num_A, int dim_A){
    int i,j;
	for (i=0;i<num_A;i++){
	    for (j=0;j<dim_A;j++){
	        A_transpose[j*num_A + i] = A[i*dim_A + j];
	    }
	} 	
}

void add_matrix(FLOAT_TYPE *A, FLOAT_TYPE *B, int n_rows, int n_cols){
    int i,j;
	for (i=0;i<n_rows;i++){
	    for (j=0;j<n_cols;j++){    
	        A[i*n_cols + j] += B[i*n_cols + j];
	    }
    }
}


void print_combination(int* combination, int k)
{
    int i;
    for (i = 0; i < k; i++){
        printf("%d ", combination[i]);
    }
    printf("\n");
}

void find_all_combinations(int idx, int* in_use, int* combination, int n, int k, int** output, int *size){
    int i;
    if (idx == k){
        //print_combination(combination, k);
        int *tmp = (int*)malloc((*size+1)*k*sizeof(int));
        memcpy(tmp, *output, *size*k*sizeof(int));        
        free(*output);
        *output = tmp;        
                        
        for (i=0; i<k; i++){
            (*output)[*size*k+i] = combination[i];
        }
                
        *size += 1;
        return;
    }

    for (i = 0; i < n; i++){
        if (in_use[i]){
            continue;
        }

        in_use[i] = 1;
        combination[idx] = i;
        idx = idx+1;

        find_all_combinations(idx, in_use, combination, n, k, output, size);
        
        idx = idx-1;
        combination[idx] = 0;
        in_use[i] = 0;
    }
}

void get_all_dimensions_to_test(int d, int K, int** all_dims, int *size){
    int *in_use = (int*)calloc(d,sizeof(int));
    int *curr_combination = (int*)calloc(K,sizeof(int));
    int *all_combinations = NULL;
    int size_all_permutations = 0;
    find_all_combinations(0, in_use, curr_combination, d, K, &all_combinations, &size_all_permutations);

    *all_dims = NULL;
    
    int num_all_dims = 0;
    int largest_first_index = -1;
    int i,j;
    for (i=0; i<size_all_permutations;i++){
        if (all_combinations[i*K+0] > largest_first_index){
            largest_first_index = all_combinations[i*K+0];
        }
        // check if one of the elements is smaller than the largest_first_index found so far        
        int include_tuple=1;
        for (j=0;j<K;j++){
            if (all_combinations[i*K+j] < largest_first_index){
                include_tuple=0;
            }
        }        
        if (include_tuple){
            int *tmp = (int*)malloc((num_all_dims+1)*K*sizeof(int));
            memcpy(tmp, *all_dims, num_all_dims*K*sizeof(int));
            free(*all_dims);
            *all_dims = tmp;

        
            for (j=0;j<K;j++){
                //printf("%d ", all_combinations[i*K+j]);
                (*all_dims)[num_all_dims*K+j] = all_combinations[i*K+j];
            }        
            num_all_dims++;            
            //printf("\n");            
        }
        
    }
    *size = num_all_dims;
    
    
}

FLOAT_TYPE get_mode_array(FLOAT_TYPE *daArray, int iSize) {
    // Allocate an int array of the same size to hold the
    // repetition count
    int* ipRepetition = (int*) calloc(iSize, sizeof(int)); //new int[iSize];
    int i;
    for (i = 0; i < iSize; ++i) {
        ipRepetition[i] = 0;
        int j = 0;
        while ((j < i) && (daArray[i] != daArray[j])) {
            if (daArray[i] != daArray[j]) {
                ++j;
            }
        }
        ++(ipRepetition[j]);
    }
    int iMaxRepeat = 0;
    for (i = 1; i < iSize; ++i) {
        if (ipRepetition[i] > ipRepetition[iMaxRepeat]) {
            iMaxRepeat = i;
        }
    }
    free(ipRepetition);
    return daArray[iMaxRepeat];
}


