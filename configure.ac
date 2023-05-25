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

# required programs
AC_PROG_CC
# AC_C__GENERIC
AC_C_RESTRICT
AC_CONFIG_FILES([Makefile])

# default prefix is "."
AC_PREFIX_DEFAULT(.)

# configure automake
AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])
AM_SILENT_RULES([yes])
PAC_ADD_CPPFLAGS([-g -Wall -std=c11])
PAC_ADD_CPPFLAGS([-Wno-deprecated-declarations])
PAC_ADD_FLAGS([-fopenmp],[-fopenmp])
PAC_ADD_LIB_PATH(fabric,[rdma/fabric.h],[fi_getinfo],[ofi])
PAC_ADD_LIB_PATH(pmi,[pmi.h],[PMI_Init],[pmi])


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

AC_OUTPUT
