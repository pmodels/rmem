# get the hostname and load the config accordingly
HOSTNAME := $(shell hostname)

#---------------------------------------------------------------------------------------------------
ifneq (,$(findstring lucia,$(HOSTNAME)))
#use cuda unless specified
USE_CUDA ?=1
# compiler and paths
CC=gcc
CXX=g++
NVCC=nvcc
PMI_DIR=${HOME}/lib-OFI-1.18.0-CUDA-11.7.0-dbg
OFI_DIR=${HOME}/lib-OFI-1.18.0-CUDA-11.7.0-dbg
#---------------------------------------------------------------------------------------------------
else
# compiler and paths
CC =clang
USE_CUDA ?=0
PMI_DIR =/Users/tgillis/dbs_lib/lib_OFI-1.18.0-dbg
OFI_DIR =/Users/tgillis/dbs_lib/lib_OFI-1.18.0-dbg
endif
