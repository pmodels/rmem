#***************************************************************************************************
#
# Copyright (C) by Argonne National Laboratory
# 	See COPYRIGHT in top-level directory
#
#***************************************************************************************************

#***************************************************************************************************
# ARCH DEPENDENT VARIABLES
ARCH_FILE ?= make_arch/default.mak
include $(ARCH_FILE)

################################################################################
# FROM HERE, DO NOT TOUCH
#-----------------------------------------------------------------------------
# Do not show GNU makefile command and info 
# .SILENT:

#-----------------------------------------------------------------------------
CC ?= gcc
CXX ?= g++
LD ?= gcc
NVCC ?= nvcc

#-----------------------------------------------------------------------------
TARGET := rmem
# git commit
# GIT_COMMIT ?= $(shell git rev-parse --short HEAD)

#-----------------------------------------------------------------------------
BUILDDIR := ./build
SRC_DIR := ./src
OBJ_DIR := ./build

## add the headers to the vpaths
INC := -I$(SRC_DIR)

#-----------------------------------------------------------------------------
#---- PMI
PMI_DIR ?= /usr
PMI_INC ?= $(PMI_DIR)/include
PMI_LIB ?= $(PMI_DIR)/lib
PMI_LIBNAME ?= -lpmi
INC += -I$(PMI_INC)
LIB += -L$(PMI_LIB) $(PMI_LIBNAME) -Wl,-rpath,$(PMI_LIB)
# LIB += -L$(PMI_LIB) $(PMI_LIBNAME) -rpath=$(PMI_LIB)

#---- OFI
OFI_DIR ?= /usr
OFI_INC ?= $(OFI_DIR)/include
OFI_LIB ?= $(OFI_DIR)/lib
OFI_LIBNAME ?= -lfabric
INC += -I$(OFI_INC)
LIB += -L$(OFI_LIB) $(OFI_LIBNAME) -Wl,-rpath,$(OFI_LIB)
# LIB += -L$(OFI_LIB) $(OFI_LIBNAME) -rpath=$(OFI_LIB)

## add the time lib
INC += -D_POSIX_C_SOURCE=199309L
INC += -pthread
LIB += -pthread -lm
LIB += -pthread -lm -lcuda -lcudart


#-----------------------------------------------------------------------------
## add the wanted folders - common folders
CC_SRC := $(notdir $(wildcard $(SRC_DIR)/*.c))
CU_SRC := $(notdir $(wildcard $(SRC_DIR)/*.cu))
CC_HEAD := $(wildcard $(SRC_DIR)/*.h)

## generate object list
CC_OBJ := $(CC_SRC:%.c=$(OBJ_DIR)/%.o)
CU_OBJ := $(CU_SRC:%.cu=$(OBJ_DIR)/%.o)
DEP := $(CC_SRC:%.c=$(OBJ_DIR)/%.d)

################################################################################
# mandatory flags
CCFLAGS ?=
GENCODE = -gencode=arch=compute_80,code=sm_80
#-fPIC -DGIT_COMMIT=\"$(GIT_COMMIT)\"   

# shenanigans
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
REAL_CU_FLAGS := $(subst $(space),$(comma),$(strip $(REAL_CC_FLAGS)))
REAL_LD_FLAGS := $(strip $(OPTS) $(LDFLAGS) $(LIB))

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -std=c11 $(REAL_CC_FLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(GENCODE) -Xcompiler $(REAL_CU_FLAGS) -c $< -o $@

# $(NVCC) $(CUFLAGS) -Xcompiler $(REAL_CU_FLAGS) -dc -c $< -o $@

$(OBJ_DIR)/$(TARGET)_dlink.o: $(CU_OBJ)
	$(NVCC) $(CUFLAGS) -dlink $^ -o $@

################################################################################
.PHONY: default
default: 
	@$(MAKE) info 
	@$(MAKE) $(TARGET)

# REAL_LD_FLAGS := $(subst $(space),$(comma),$(strip $(OPTS) $(LDFLAGS) $(LIB)))
REAL_LD_FLAGS := $(strip $(OPTS) $(LDFLAGS) $(LIB))
$(TARGET):$(CC_OBJ) $(CU_OBJ)
	$(CC) $(REAL_LD_FLAGS) $^ -o $@

# $(NVCC) $(CUFLAGS) -Xlinker $(REAL_LD_FLAGS) $^ -o $@

# $(TARGET): $(CC_OBJ) $(OBJ_DIR)/$(TARGET)_dlink.o
# 	$(CC) $(OPTS) $(LDFLAGS) $(LIB) $^ -o $@

################################################################################
.PHONY: debug
debug:
	@OPTS="-O0 -g -fsanitizer" $(MAKE) $(TARGET)
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
