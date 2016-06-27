/* 
 * loss_functions.c
 */
#include "loss_functions.h"

FLOAT_TYPE get_loss(FLOAT_TYPE *labels, int num_labels, FLOAT_TYPE *predictions, int num_predictions, int loss_function){
    FLOAT_TYPE loss;
    switch(loss_function){
        case RMSE_NORM:
            loss = rmse_norm_loss(labels, predictions, num_labels);
            break;
        case RMSE:
            loss = rmse_loss(labels, predictions, num_labels);
            break;
        case ZERO_ONE:
            loss = zero_one_loss(labels, predictions, num_labels);
            break;
        case MAD:
            loss = mad_loss(labels, predictions, num_labels);
            break;
        case MAD_NORM:
            loss = mad_norm_loss(labels, predictions, num_labels);                        
            break;
        case STD:
            loss = std_loss(labels, predictions, num_labels);
            break;
        case RMSE_NORM_MAD_NORM_STD:
            loss = rmse_norm_mad_norm_std_loss(labels, predictions, num_labels);   
            break;
        case RMSE_NORM_MAD_NORM_GALAXIES:
            loss = rmse_norm_mad_norm_loss_galaxies(labels, predictions, num_labels);   
            break;
        case RMSE_STD:
            loss = rmse_loss(labels, predictions, num_labels) + std_loss(labels, predictions, num_labels);
            break;
    }
    return loss;
}


FLOAT_TYPE rmse_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    int i;
    FLOAT_TYPE sum = 0.0;
    for (i=0;i<num;i++){
        FLOAT_TYPE tmp = (labels[i] - predictions[i]);
        sum += tmp*tmp;
    }
    sum /= num;
    return sqrt(sum);
}

FLOAT_TYPE rmse_norm_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    int i;
    FLOAT_TYPE sum = 0.0;
    for (i=num;i--;){
        FLOAT_TYPE target_val = labels[i];
        FLOAT_TYPE tmp = ((target_val-predictions[i])/(1.0+target_val));
        sum += tmp*tmp;
    }
    sum /= num;
    return sqrt(sum);
}




FLOAT_TYPE mad_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    int i;
    FLOAT_TYPE *all_errors = (FLOAT_TYPE*)malloc(num*sizeof(FLOAT_TYPE));
    // compute all errors
    for (i=0;i<num;i++){
        FLOAT_TYPE target_val = labels[i];
        FLOAT_TYPE pred = predictions[i];        
        FLOAT_TYPE error = (target_val-pred);
        all_errors[i] = error;    
    }
    // find median
    FLOAT_TYPE median1 = find_median(all_errors,0,num);
    // subtract
    for (i=0;i<num;i++){
        all_errors[i] = fabs(all_errors[i] - median1);
    }
    // find median
    FLOAT_TYPE median2 = find_median(all_errors,0,num);    

    free(all_errors);
    return median2;
}

FLOAT_TYPE mad_norm_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    int i;
    FLOAT_TYPE *all_errors = (FLOAT_TYPE*)malloc(num*sizeof(FLOAT_TYPE));
    // compute all errors
    for (i=0;i<num;i++){
        FLOAT_TYPE target_val = labels[i];
        FLOAT_TYPE pred = predictions[i];        
        FLOAT_TYPE error = (target_val-pred)/(1.0+target_val);
        all_errors[i] = error;    
    }
    // find median
    FLOAT_TYPE median1 = find_median(all_errors,0,num);
    // subtract
    for (i=0;i<num;i++){
        all_errors[i] = fabs(all_errors[i] - median1);
    }
    // find median
    FLOAT_TYPE median2 = find_median(all_errors,0,num);    

    free(all_errors);
    return median2;
}

FLOAT_TYPE zero_one_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    int i;
    FLOAT_TYPE sum = 0.0;
    for (i=0;i<num;i++){
        FLOAT_TYPE target_val = labels[i];
        FLOAT_TYPE pred = predictions[i];
        if (abs(target_val-pred)<10E-8){
            sum += 0.0;
        } else {
            sum += 1.0;
        }  
    }
    sum /= num;
    return sum;
}

FLOAT_TYPE std_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    int i;
    FLOAT_TYPE mean = 0.0;
    for (i=0;i<num;i++){
        mean += (labels[i]-predictions[i]);
    }
    mean /= num;

    FLOAT_TYPE sum = 0.0;
    for (i=0;i<num;i++){
        FLOAT_TYPE target_val = labels[i];
        FLOAT_TYPE pred = predictions[i];
        sum = ((target_val-pred) - mean)*((target_val-pred) - mean);
    }
    sum /= (num-1);
    return sqrt(sum);
}

FLOAT_TYPE rmse_norm_mad_norm_std_loss(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    FLOAT_TYPE rmse_norm = rmse_norm_loss(labels, predictions, num);
    FLOAT_TYPE mad_norm = mad_norm_loss(labels, predictions, num);
    FLOAT_TYPE std = std_loss(labels, predictions, num);
    return rmse_norm/0.19 + mad_norm/0.041 + std/0.16;
}

FLOAT_TYPE rmse_norm_mad_norm_loss_galaxies(FLOAT_TYPE *labels, FLOAT_TYPE *predictions, int num){
    FLOAT_TYPE rmse_norm = rmse_norm_loss(labels, predictions, num);
    FLOAT_TYPE mad_norm = mad_norm_loss(labels, predictions, num);
    FLOAT_TYPE std = std_loss(labels, predictions, num);
    return rmse_norm/0.019 + mad_norm/0.009;
}

//////////////////////////////////////////////////////////////////////////////////
int cmpfunc (const void * a, const void * b)
{

    if (*(FLOAT_TYPE*)a - *(FLOAT_TYPE*)b < 0)
        return -1;
    if (*(FLOAT_TYPE*)a - *(FLOAT_TYPE*)b > 0)
        return +1;        
    return 0;
} 


FLOAT_TYPE find_median(FLOAT_TYPE* errors, int start, int end){
    FLOAT_TYPE *errors_sorted = (FLOAT_TYPE*)malloc((end-start)*sizeof(FLOAT_TYPE));
    int i;
    for (i=0;i<end-start;i++){
        errors_sorted[i] = errors[start+i];
    }
    qsort(errors_sorted, end-start, sizeof(FLOAT_TYPE), cmpfunc);
    FLOAT_TYPE median = errors_sorted[(int)(end-start)/2];
    free(errors_sorted);
    return median;
}


