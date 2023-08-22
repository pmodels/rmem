# get the hostname and load the config accordingly
HOSTNAME := $(shell hostname)

#---------------------------------------------------------------------------------------------------
# LUCIA
ifneq (,$(findstring lucia,$(HOSTNAME)))
CC=gcc
CXX=g++
PMI_DIR=${HOME}/lib-OFI-1.18.0
OFI_DIR=${HOME}/lib-OFI-1.18.0
#---------------------------------------------------------------------------------------------------
# LUMI
else ifneq (,$(or $(findstring uan,$(HOSTNAME)),$(findstring nid,$(HOSTNAME))))
CC=gcc
CXX=g++
PMI_DIR=${HOME}/lib-PMI-4.1.1
OFI_DIR=/opt/cray/libfabric/1.15.2.0
OFI_LIB=/opt/cray/libfabric/1.15.2.0/lib64
#---------------------------------------------------------------------------------------------------
else
CC=clang
PMI_DIR=/Users/tgillis/dbs_lib/lib_OFI-1.18.1-dbg
OFI_DIR=/Users/tgillis/dbs_lib/lib_OFI-1.18.1-dbg
ARGP_DIR=/opt/homebrew
endif
