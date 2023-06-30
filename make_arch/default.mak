# get the hostname and load the config accordingly
HOSTNAME := $(shell hostname)

#---------------------------------------------------------------------------------------------------
ifneq (,$(findstring lucia,$(HOSTNAME)))
CC=gcc
CXX=g++
NVCC=nvcc
PMI_DIR=${HOME}/lib-OFI-1.18.0-CUDA-11.7.0-dbg
OFI_DIR=${HOME}/lib-OFI-1.18.0-CUDA-11.7.0-dbg
#---------------------------------------------------------------------------------------------------
else
CC=clang
PMI_DIR=/Users/tgillis/dbs_lib/lib_OFI-1.18.0-dbg
OFI_DIR=/Users/tgillis/dbs_lib/lib_OFI-1.18.0-dbg
endif
