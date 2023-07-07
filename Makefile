#***************************************************************************************************
#
# Copyright (C) by Argonne National Laboratory
# 	See COPYRIGHT in top-level directory
#
#***************************************************************************************************

#***************************************************************************************************
# ARCH DEPENDENT VARIABLES
ARCH_FILE ?= make_arch/default.mak

####################################################################################################
# FROM HERE, DO NOT TOUCH
include $(ARCH_FILE)
# Do not show GNU makefile command and info 
#.SILENT:

#---------------------------------------------------------------------------------------------------
CC ?= gcc
CXX ?= g++
LD ?= gcc
NVCC ?= nvcc

#---------------------------------------------------------------------------------------------------
TARGET := rmem

BUILDDIR := ./build
SRC_DIR := ./src
OBJ_DIR := ./build
# git commit
# GIT_COMMIT ?= $(shell git rev-parse --short HEAD)

## add the headers to the vpaths
INC := -I$(SRC_DIR)

#---------------------------------------------------------------------------------------------------
# PMI
PMI_DIR ?= /usr
PMI_INC ?= $(PMI_DIR)/include
PMI_LIB ?= $(PMI_DIR)/lib
PMI_LIBNAME ?= -lpmi

# OFI
OFI_DIR ?= /usr
OFI_INC ?= $(OFI_DIR)/include
OFI_LIB ?= $(OFI_DIR)/lib
OFI_LIBNAME ?= -lfabric

#---------------------------------------------------------------------------------------------------
# includes
INC += -I$(PMI_INC)
INC += -I$(OFI_INC)
# pthread 
INC += -pthread
# gcc need this special define to handle time measurement
ifneq (,$(findstring gcc,$(CC)))
INC += -D_POSIX_C_SOURCE=199309L
endif

# add the link options
LIB += -lpthread -lm
LIB += -L$(PMI_LIB) $(PMI_LIBNAME)
LIB += -L$(OFI_LIB) $(OFI_LIBNAME)
# different way of doing rpath
ifeq ($(USE_CUDA),1)
LIB += -rpath=$(PMI_LIB)
LIB += -rpath=$(OFI_LIB)
else
LIB += -Wl,-rpath,$(PMI_LIB)
LIB += -Wl,-rpath,$(OFI_LIB)
endif

#---------------------------------------------------------------------------------------------------
## add the wanted folders - common folders
CC_SRC := $(notdir $(wildcard $(SRC_DIR)/*.c))
CU_SRC := $(notdir $(wildcard $(SRC_DIR)/*.cu))
CC_HEAD := $(wildcard $(SRC_DIR)/*.h)

## generate object list
CC_OBJ := $(CC_SRC:%.c=$(OBJ_DIR)/%.o)
CU_OBJ := $(CU_SRC:%.cu=$(OBJ_DIR)/%.o)
DEP := $(CC_SRC:%.c=$(OBJ_DIR)/%.d)

#create the object list
OBJ := $(CC_OBJ) 
# add the cuda objects
ifeq ($(USE_CUDA),1)
OBJ += $(CU_OBJ)
endif

################################################################################
# mandatory flags
CCFLAGS =
ifeq ($(USE_CUDA),1)
CCFLAGS += -DHAVE_CUDA
endif

#-fPIC -DGIT_COMMIT=\"$(GIT_COMMIT)\"   
GENCODE = -gencode=arch=compute_80,code=sm_80

# Makefile shenanigans
comma:= ,
empty:=
space:= $(empty) $(empty)

################################################################################
.PHONY: default
default: 
	@$(MAKE) info 
	@$(MAKE) $(TARGET)

################################################################################
# get the full list of flags for CC
REAL_CC_FLAGS:=$(strip $(OPTS) $(CCFLAGS) $(INC))
REAL_CU_FLAGS:=$(subst $(space),$(comma),$(strip $(REAL_CC_FLAGS)))

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "$(CC) $@"
	$(CC) -std=c11 $(REAL_CC_FLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "$(NVCC) $@"
	$(NVCC) $(GENCODE) -Xcompiler $(REAL_CU_FLAGS) -c $< -o $@

#---------------------------------------------------------------------------------------------------
# link stage
ifeq ($(USE_CUDA),1)
# we use cuda to link
LD := $(NVCC) $(GENCODE)
REAL_LD_FLAGS := -Xlinker $(subst $(space),$(comma),$(strip $(LDFLAGS) $(LIB)))
ifneq (,$(OPTS))
# the options must be given as a compiler flag (-fsanitize etc)
REAL_LD_FLAGS += -Xcompiler $(subst $(space),$(comma),$(strip $(OPTS)))
endif

else
# use normal GCC to link
LD := $(CC)
REAL_LD_FLAGS:= $(strip $(OPTS) $(LDFLAGS) $(LIB))
endif
# link recipe
$(TARGET):$(OBJ)
	@echo "$(LD) $@"
	$(LD) $(REAL_LD_FLAGS) $^ -o $@

################################################################################
.PHONY: debug
debug:
	@OPTS="-O0 -g -fsanitize=address -fsanitize=undefined" $(MAKE) $(TARGET)
.PHONY: fast
fast:
	@OPTS="-O3 -DNEBUG" $(MAKE) $(TARGET)
################################################################################
clean:
	@rm -f $(TARGET)_dlink.o
	@rm -f $(OBJ_DIR)/*.o
	@rm -f $(TARGET)

reallyclean:
	$(MAKE) clean
	@rm -f $(OBJ_DIR)/*.d
	@rm -rf rmem

#-------------------------------------------------------------------------------
# mentioning this target will export all the current variables to child-make processes
.EXPORT_ALL_VARIABLES:
.PHONY: info
info: 
	@$(MAKE) --file=make_arch/info.mak
#-------------------------------------------------------------------------------

-include $(DEP)
