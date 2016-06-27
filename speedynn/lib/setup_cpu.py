#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# get absolute kernel path
current_path = os.path.abspath( __file__ ).split("setup_cpu.py")[0]
kernel_path =  current_path + "kernels/"

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# nearest_neighbors_kdtree extension module
_nearest_neighbors_feature_selection_cpu = Extension("_nearest_neighbors_feature_selection_cpu",
                   ["nearest_neighbors_feature_selection_float_cpu.i","nearest_neighbors_feature_selection.c","util.c","cpu.c","gpu.c","loss_functions.c"],
                   include_dirs = [numpy_include,"/home/fgieseke/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc"],
                   define_macros=[
                   ('FLOAT_TYPE', "float"),
                   ('WORKGROUP_SIZE', 128),
                   ('WORKGROUP_SIZE_SMALLEST_VALIDATION_ERROR', 256),
                   ('USE_LOCAL_MEM',0),
                   ('DEBUG', 1),
                   ('USE_GPU', 0),
                   ('GPU_PLATFORM_NUMBER', 0),        
                   ('KERNEL_NAME_COMPUTE_DISTANCE_MATRIX', "\"" + kernel_path + "compute_distance_matrix_float.cl" + "\""),
                   ('KERNEL_NAME_COMPUTE_DISTANCE_MATRIX_SELECTED', "\"" + kernel_path + "compute_distance_matrix_selected_float.cl" + "\""),
                   ('KERNEL_NAME_UPDATE_CURRENT_DISTANCE_MATRIX', "\"" + kernel_path + "update_current_distance_matrix_float.cl" + "\""),
                   ('KERNEL_NAME_GET_SMALLEST_VALIDATION_ERROR', "\"" + kernel_path + "get_smallest_validation_error_float.cl" + "\""),
                   ('KERNEL_NAME_COMPUTE_PREDICTIONS', "\"" + kernel_path + "compute_predictions.cl" + "\"")
                   ],                 
                   libraries=['OpenCL']
                   )

# NumyTypemapTests setup
setup(  name        = "nearest_neighbors_feature_selection_cpu",
        description = "nearest_neighbors_feature_selection_cpu",

        author      = "Fabian Gieseke",
        version     = "0.1",
        ext_modules = [_nearest_neighbors_feature_selection_cpu]
        )
