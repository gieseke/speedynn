/* 
 * cpu.h
 */
#ifndef CPU
#define CPU

#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "loss_functions.h"
#include "extern.h"
#include "util.h"

#include "nearest_neighbors_feature_selection.h"

void init_memory_cpu(int large_scale);
void free_memory_cpu(int large_scale);

void get_k_nearest_neighbors_cpu(int train_idx, int fold, int num_per_fold, int *k_max_nn_indices, FLOAT_TYPE *k_max_nn_distances, int dim_removed, int mult);
void get_smallest_validation_errors_cpu(int *selected_dimensions, FLOAT_TYPE *val_errors, int mult, int large_scale, int *loss_functions, int num_loss_functions, int model_type);


void empty_distance_matrix_cpu(void);
void compute_distance_matrix_cpu(int *selected_dimensions, int selected_only);
void update_current_dist_matrix_cpu(int min_dim, int mult);
#endif
