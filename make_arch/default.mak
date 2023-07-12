# get the hostname and load the config accordingly
HOSTNAME := $(shell hostname)

#---------------------------------------------------------------------------------------------------
ifneq (,$(findstring lucia,$(HOSTNAME)))
CC=gcc
CXX=g++
PMI_DIR=${HOME}/lib-OFI-1.18.0
OFI_DIR=${HOME}/lib-OFI-1.18.0
#---------------------------------------------------------------------------------------------------
else
CC=clang
PMI_DIR=/Users/tgillis/dbs_lib/lib_OFI-1.18.0-dbg
OFI_DIR=/Users/tgillis/dbs_lib/lib_OFI-1.18.0-dbg
endif
