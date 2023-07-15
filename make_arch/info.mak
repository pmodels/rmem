# Prevent the parallel execution when calling this Makefile
.NOTPARALLEL:

.PHONY: info
info: logo
	$(info using $(ARCH_FILE) on $(HOSTNAME):)
	$(info compiler: $(CC) $(shell $(CC) -dumpversion))
	$(info opts    : $(OPTS))
	$(info ------------)
	$(info PMI:)
	$(info - include: -I$(PMI_INC) )
	$(info - lib: -L$(PMI_LIB) $(PMI_LIBNAME) -Wl,-rpath,$(PMI_LIB))
	$(info ------------)
	$(info OFI:)
	$(info - include: -I$(OFI_INC) )
	$(info - lib: -L$(OFI_LIB) $(OFI_LIBNAME) -Wl,-rpath,$(OFI_LIB))
	$(info ------------)
	$(info CC flags: $(REAL_CC_FLAGS))
	$(info LD flags: $(REAL_LD_FLAGS))
	$(info ----------------------------------------------)

.NOTPARALLEL: logo
.PHONY: logo
logo: 
	@echo "----------------------------------------------"
	@echo "                                              "
	@echo "    ██████╗ ███╗   ███╗███████╗███╗   ███╗    "
	@echo "    ██╔══██╗████╗ ████║██╔════╝████╗ ████║    "
	@echo "    ██████╔╝██╔████╔██║█████╗  ██╔████╔██║    "
	@echo "    ██╔══██╗██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║    "
	@echo "    ██║  ██║██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║    "
	@echo "    ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝    "
	@echo "                                              "
	@echo "          (C) Argonne National Lab            "
	@echo "                                              "
	@echo "----------------------------------------------"



