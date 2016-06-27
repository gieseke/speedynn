/* 
 * util.h
 */
#ifndef UTIL
#define UTIL

#include "types.h"
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>


#define TIMING


/* -------------------------------------------------------------------------------- 
 * Computes the distance between point a and b in R^dim
 * -------------------------------------------------------------------------------- 
*/
FLOAT_TYPE dist(FLOAT_TYPE *a, FLOAT_TYPE *b, int dim);

FLOAT_TYPE dist_dimension(FLOAT_TYPE *a, FLOAT_TYPE *b, int dim_selected);

int get_max_value(int *array, int size_array);

/*
 * Helper method for computing the current time (w.r.t to an offset).
 */
long get_system_time_in_microseconds(void);


void matrix_multiply(FLOAT_TYPE *A, FLOAT_TYPE *B, FLOAT_TYPE *C, int A_n_rows, int A_n_cols, int B_n_cols, FLOAT_TYPE multiplier);
void compute_transpose(FLOAT_TYPE *A, FLOAT_TYPE *A_transpose, int num_A, int dim_A);
void add_matrix(FLOAT_TYPE *A, FLOAT_TYPE *B, int n_rows, int n_cols);

FLOAT_TYPE get_mode_array(FLOAT_TYPE *daArray, int iSize);
/* -------------------------------------------------------------------------------- 
 * Timing-related macros.
 * -------------------------------------------------------------------------------- 
 */
#ifdef TIMING
// Declare variables for timer 'number'
#define DEFINE_TIMER(number)						\
    long timer_start##number  = 0.0f;				\
    long timer_stop##number   = 0.0f;				\
    long timer_length##number = 0.0f;				\
    double  timer_sum##number    = 0.0f;

#define DECLARE_TIMER(number)						\
    extern long timer_start##number  ;				\
    extern long timer_stop##number   ;				\
    extern long timer_length##number ;				\
    extern double  timer_sum##number ;


// Reset and start timer 'number'
#define START_TIMER(number)						\
    timer_start##number  = get_system_time_in_microseconds();		\
    timer_length##number = 0.0f;

// Pause timer 'number'
#define STOP_TIMER(number)						\
    timer_stop##number = get_system_time_in_microseconds();		\
    timer_length##number +=						\
	      (((double)timer_stop##number)-((double)timer_start##number)); \
    timer_start##number = get_system_time_in_microseconds();

// Resume timer 'number'
#define RESUME_TIMER(number)			\
    timer_length##start = get_system_time_in_microseconds();

// Print the recorded time (between the last reset and the last stop) of timer 'number'
#define PRINT_TIMER(number)						\
    printf("Elapsed time in timer %i: %3.20f sec.\n", number, (double)(1.0*timer_length##number/ 1000000.0));

// Add the recorded time of timer 'number' to a global counter for timer 'number'
#define RECORD_TIMER_SUM(number)		\
    timer_sum##number += timer_length##number;

// Add the recorded time of timer 'number' to a global counter for timer 'number'
#define RESET_TIMER_SUM(number)		\
    timer_sum##number = 0.0f;

// Print the time recorded by the global counter for timer 'number' averaged over 'numit'.
#define PRINT_AVERAGE_TIMER(number, numit)				\
    printf("Average cost per iteration: %f sec.\n",			\
	   (double)(1.0 / 1000000.0)*(double)timer_sum##number / (double)numit);

#define TIMER_VALUE(number) timer_length##number
#define TIMER_SUM(number) timer_sum##number

#else  // TIMING undefined
#define DEFINE_TIMER(number) 
#define DECLARE_TIMER(number)
#define START_TIMER(number) 
#define STOP_TIMER(number) 
#define RESUME_TIMER(number) 
#define PRINT_TIMER(number) 
#define RECORD_TIMER_SUM(number)
#define RESET_TIMER_SUM(number)
#define PRINT_AVERAGE_TIMER(number, numit)
#define TIMER_VALUE(number) 
#define TIMER_SUM(number) 
#endif  // TIMING

#endif
