/* 
 * types.h
 */
#ifndef TYPES
#define TYPES

#if FLOAT_TYPE == float 
    #define MAX_FLOAT_TYPE     3.402823466e+38
    #define MIN_FLOAT_TYPE     -3.402823466e+38
#else
    #define MAX_FLOAT_TYPE     1.7976931348623158e+308 
    #define MIN_FLOAT_TYPE     -1.7976931348623158e+308 
#endif

#endif


