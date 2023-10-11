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
.SILENT:

USE_CUDA?=0
USE_HIP?=0
#---------------------------------------------------------------------------------------------------
CC ?= gcc
CXX ?= g++
LD ?= gcc
NVCC ?= nvcc
HIPCC ?= hipcc

#---------------------------------------------------------------------------------------------------
TARGET := rmem

BUILDDIR := ./build
SRC_DIR := ./src
OBJ_DIR := ./build
# git commit
GIT_COMMIT ?= $(shell git rev-parse --short HEAD)

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

# ARGP
ARGP_DIR ?= /usr
ARGP_INC ?= $(ARGP_DIR)/include
ARGP_LIB ?= $(ARGP_DIR)/lib
ARGP_LIBNAME ?= -largp

#---------------------------------------------------------------------------------------------------
# includes
INC += -I$(OFI_INC)
INC += -I$(PMI_INC)
# pthread 
INC += -pthread
# gcc need this special define to handle time measurement
ifneq (,$(findstring gcc,$(CC)))
INC += -D_POSIX_C_SOURCE=199309L
endif

# add the link options
LIB =
LIB += -lpthread -lm
LIB += -L$(OFI_LIB) $(OFI_LIBNAME)
LIB += -L$(PMI_LIB) $(PMI_LIBNAME)
# different way of doing rpath in CUDA
ifeq ($(USE_CUDA),1)
LIB += -rpath=$(OFI_LIB)
LIB += -rpath=$(PMI_LIB)
else
LIB += -Wl,-rpath,$(OFI_LIB)
LIB += -Wl,-rpath,$(PMI_LIB)
endif

# if not gcc, add argp lib
ifeq (,$(findstring gcc,$(CC)))
INC += -I$(ARGP_INC)
LIB += -L$(ARGP_LIB) $(ARGP_LIBNAME)
LIB += -Wl,-rpath,$(ARGP_LIB)
endif

#---------------------------------------------------------------------------------------------------
## add the wanted folders - common folders
CC_SRC := $(notdir $(wildcard $(SRC_DIR)/*.c))
CU_SRC := $(notdir $(wildcard $(SRC_DIR)/*.cu))
HIP_SRC := $(notdir $(wildcard $(SRC_DIR)/*.hip))
CC_HEAD := $(wildcard $(SRC_DIR)/*.h)

## generate object list
CC_OBJ := $(CC_SRC:%.c=$(OBJ_DIR)/%.o)
CU_OBJ := $(CU_SRC:%.cu=$(OBJ_DIR)/%.o)
HIP_OBJ := $(HIP_SRC:%.hip=$(OBJ_DIR)/%.o)
DEP := $(CC_SRC:%.c=$(OBJ_DIR)/%.d)

#create the object list
OBJ := $(CC_OBJ) 
# add the cuda objects
ifeq ($(USE_CUDA),1)
OBJ += $(CU_OBJ)
$(info cuda OBJ = ${OBJ})
else ifeq ($(USE_HIP),1)
$(info hip OBJ = ${OBJ})
OBJ += $(HIP_OBJ)
$(info hip OBJ = ${OBJ})
endif

################################################################################
# mandatory flags
CCFLAGS ?=
CCFLAGS += -Wno-deprecated-declarations -Wshadow
ifneq (,$(GIT_COMMIT))
CCFLAGS += -DGIT_COMMIT=\"$(GIT_COMMIT)\"   
endif

GENCODE ?=
ifeq ($(USE_CUDA),1)
GENCODE += -gencode=arch=compute_80,code=sm_80
CCFLAGS += -DHAVE_CUDA
else ifeq ($(USE_HIP),1)
#GENCODE += -offload-arch=auto
CCFLAGS += -DHAVE_HIP
#capture the includes for HIP
CCFLAGS += $(shell hipconfig --cpp_config)
endif


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

ifeq ($(USE_CUDA),1)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "cuda: $(NVCC) $@"
	$(NVCC) $(GENCODE) -Xcompiler $(REAL_CU_FLAGS) -c $< -o $@
else ifeq ($(USE_HIP),1)
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.hip
	@echo "hip: $(HIPCC) $@"
	$(HIPCC) $(GENCODE) -std=c++11 $(REAL_CC_FLAGS) -MMD -c $< -o $@
endif
#---------------------------------------------------------------------------------------------------
# link stage
# ======= CUDA =======
ifeq ($(USE_CUDA),1)
# we use cuda to link
LD := $(NVCC) -lcuda $(GENCODE)
REAL_LD_FLAGS := -Xlinker $(subst $(space),$(comma),$(strip $(LDFLAGS) $(LIB)))
# pass the opts to the compiler first
ifneq (,$(OPTS))
REAL_LD_FLAGS += -Xcompiler $(subst $(space),$(comma),$(strip $(OPTS)))
endif
# ======= HIP =======
else ifeq ($(USE_HIP),1)
LD := $(HIPCC)
$(info using LD = ${LD})
REAL_LD_FLAGS:= $(strip $(OPTS) $(LDFLAGS) $(LIB))
# ======= no GPU =======
else
# use normal GCC to link
LD := $(CC) $(GENCODE)
REAL_LD_FLAGS:= $(strip $(OPTS) $(LDFLAGS) $(LIB))
endif



# link recipe
$(TARGET):$(OBJ)
	@echo "$(LD) $@"
	$(LD) $(REAL_LD_FLAGS) $^ -o $@

################################################################################
.PHONY: debug
debug:
	@OPTS="${OPTS} -O0 -g -fsanitize=address -fsanitize=undefined" $(MAKE) $(TARGET)
.PHONY: verbose
verbose:
	@OPTS="${OPTS} -DVERBOSE" $(MAKE) debug
.PHONY: fast
fast:
	@OPTS="${OPTS} -O3 -DNDEBUG" $(MAKE) $(TARGET)
################################################################################
clean:
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
