# Prevent the parallel execution when calling this Makefile
.NOTPARALLEL:

.PHONY: info
info: logo
	$(info compiler: $(CC) $(shell $(CC) -dumpversion))
	$(info linker: $(LD))
	$(info cc flags: $(CCFLAGS))
	$(info opts    : $(OPTS))
	$(info ld flags: $(LDFLAGS))
	$(info using $(ARCH_FILE) )
	$(info ------------)
	$(info PMI:)
	$(info - include: -I$(PMI_INC) )
	$(info - lib: -L$(PMI_LIB) $(PMI_LIBNAME) -Wl,-rpath,$(PMI_LIB))
	$(info ------------)
	$(info OFI:)
	$(info - include: -I$(OFI_INC) )
	$(info - lib: -L$(OFI_LIB) $(OFI_LIBNAME) -Wl,-rpath,$(OFI_LIB))
	$(info ------------)

.NOTPARALLEL: logo
.PHONY: logo
logo: 
	@echo "-----------------------------------------"
	@echo "       _____  __  __ ______ __  __  "
	@echo "      |  __ \|  \/  |  ____|  \/  | "
	@echo "      | |__) | \  / | |__  | \  / | "
	@echo "      |  _  /| |\/| |  __| | |\/| | "
	@echo "      | | \ \| |  | | |____| |  | | "
	@echo "      |_|  \_\_|  |_|______|_|  |_| "
	@echo "                                    "
	@echo "        (C) Argonne National Lab    "
	@echo "                                    "
	@echo "----------------------------------------------------"



