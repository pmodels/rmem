# get the hostname and load the config accordingly
HOSTNAME := $(shell hostname)

#---------------------------------------------------------------------------------------------------
ifneq (,$(findstring lucia,$(HOSTNAME)))
#use cuda unless specified otherwise
USE_CUDA ?=1
# compiler and paths
CC=gcc
CXX=g++
NVCC=nvcc
PMI_DIR?=${HOME}/lib-OFI-1.18.0-CUDA-11.7.0-dbg
OFI_DIR?=${HOME}/lib-OFI-1.18.0-CUDA-11.7.0-dbg
#---------------------------------------------------------------------------------------------------
# LUMI
else ifneq (,$(or $(findstring uan,$(HOSTNAME)),$(findstring nid,$(HOSTNAME))))
CC=gcc
CXX=g++
PMI_DIR?=${HOME}/lib-PMI-4.1.1
OFI_DIR?=/opt/cray/libfabric/1.15.2.0
OFI_LIB?=/opt/cray/libfabric/1.15.2.0/lib64
#---------------------------------------------------------------------------------------------------
# MELUXINA
else ifneq (,$(findstring mel,$(HOSTNAME)))
CC=gcc
CXX=g++
PMI_DIR?=${HOME}/lib-OFI-1.18.1-CUDA-11.7.0
#OFI_DIR=${HOME}/libfabric/_inst
OFI_DIR?=${HOME}/lib-OFI-1.18.1-CUDA-11.7.0
#---------------------------------------------------------------------------------------------------
else
USE_CUDA ?=0
CC=clang
PMI_DIR?=/Users/tgillis/dbs_lib/lib_OFI-1.18.1-dbg
OFI_DIR?=/Users/tgillis/dbs_lib/lib_OFI-1.18.1-dbg
ARGP_DIR?=/opt/homebrew
endif
