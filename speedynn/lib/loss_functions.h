/* 
 * loss_functions.h
 */
#ifndef LOSS_FUNCTIONS
#define LOSS_FUNCTIONS

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RMSE 0
#define RMSE_NORM 1
#define MAD 2
#define MAD_NORM 3
#define ZERO_ONE 4
#define STD 5
#define RMSE_NORM_MAD_NORM_STD 6
#define RMSE_NORM_MAD_NORM_GALAXIES 7
#define RMSE_STD 8

FLOAT_TYPE get_loss(FLOAT_TYPE *labels, int num_labels, FLOAT_TYPE *predictions, int num_predictions, int loss_function);


FLOAT_TYPE rmse_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);
FLOAT_TYPE rmse_norm_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);

FLOAT_TYPE mad_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);
FLOAT_TYPE mad_norm_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);

FLOAT_TYPE zero_one_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);

FLOAT_TYPE std_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);

FLOAT_TYPE rmse_norm_mad_norm_std_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);
FLOAT_TYPE rmse_norm_mad_norm_loss_galaxies(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num);

FLOAT_TYPE find_median(FLOAT_TYPE* errors, int start, int end);
#endif

