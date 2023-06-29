#
# Copyright (C) by Argonne National Laboratory
#  See COPYRIGHT in top-level directory
#

bin_PROGRAMS = rmem

rmem_SOURCES =
include src/Makefile.mk

# How to compile .cu files
.cu.o:
	$(NVCC) -arch=native $(AM_CPPFLAGS) -c -o $@ $<

