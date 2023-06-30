# 
# Copyright (C) by Argonne National Laboratory
#  See COPYRIGHT in top-level directory
# 
AC_PREREQ([2.69])
AC_INIT([rmem],[0.1],[https://github.com/pmodels/rmem])


# use .aux folder for all the local files
AC_CONFIG_AUX_DIR([.aux])
AC_CONFIG_MACRO_DIR([m4])
AC_USE_SYSTEM_EXTENSIONS

# unique file to the source folder
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_HEADERS([.aux/rmem_config.h])

# required programs - C
AC_PROG_CC
AC_LANG([C])
AC_C_RESTRICT
AC_CONFIG_FILES([Makefile])

# default prefix is "."
AC_PREFIX_DEFAULT(.)

# configure automake
AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])
AM_SILENT_RULES([yes])
PAC_ADD_CFLAGS([-std=c11])
PAC_ADD_CPPFLAGS([-g -Wall])
PAC_ADD_CPPFLAGS([-Wno-deprecated-declarations])
PAC_ADD_FLAGS([-pthread],[],[-pthread])
PAC_ADD_FLAGS([-lm],[],[-lm])
# requirement functions
AC_C_INLINE
AC_FUNC_MALLOC
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT8_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_CHECK_FUNCS([clock_gettime memset mkdir pow sqrt])
AC_CHECK_HEADERS([limits.h stddef.h])
AC_CHECK_HEADER_STDBOOL
AC_CHECK_TYPES([atomic_int], [], [], [[#include <stdatomic.h>]])

# dependencies
PAC_ADD_LIB_PATH(fabric,[rdma/fabric.h],[fi_getinfo],[ofi],[])
PAC_ADD_LIB_PATH(pmi,[pmi.h],[PMI_Init],[pmi],[])


PAC_ADD_ENABLE_MODE([verbose],
                    [-DVERBOSE],
                    [],
                    [enable verbose mode])
PAC_ADD_ENABLE_MODE([fast],
                    [-O3 -DNDEBUG],
                    [-O3 -flto],
                    [enable fastest code, conflicts with --enable-asan])
PAC_ADD_ENABLE_MODE([asan],
                    [-g -pg -O0 -fsanitize=address],
                    [-fsanitize=address],
                    [enable address sanitizer, conflicts with --enable-fast])

# add cuda specific treatment
AC_ARG_VAR([NVCC], [nvcc compiler to use, see --with-cuda])
AC_SUBST([NVCC],[nvcc])
PAC_ADD_LIB_PATH(cudart,[cuda.h],[cudaGetDevice],[cuda],[])


AC_OUTPUT
