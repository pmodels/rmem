#
# Copyright (C) by Argonne National Laboratory
#  See COPYRIGHT in top-level directory
#

bin_PROGRAMS = rmem

rmem_SOURCES =
include src/Makefile.mk

# How to compile .cu files
.cu.o:
	$(NVCC) -arch=native -forward-unknown-opts $(CPPFLAGS) $(LDFLAGS) $(LIBS) -c -o $@ $<
# $(NVCC) -arch=native -forward-unknown-opts -ccbin "./mycc.sh" $(CPPFLAGS) -c -o $@ $<
# $(NVCC) -arch=native -ccbin g++  $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<
# $(NVCC) -arch=native -forward-unknown-to-host-compiler $(CPPFLAGS) -c -o $@ $<

