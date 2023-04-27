AC_PREREQ([2.71])
AC_INIT([trigr],[0.1],[https://github.com/thomasgillis/trigr])


# use .aux folder for all the local files
AC_CONFIG_AUX_DIR([.aux])
AC_CONFIG_MACRO_DIR([m4])
AC_USE_SYSTEM_EXTENSIONS

# unique file to the source folder
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_HEADERS([.aux/trigr_config.h])

# required programs
AC_PROG_CC
AC_C__GENERIC
AC_C_RESTRICT
AC_CONFIG_FILES([Makefile])

# default prefix is "."
AC_PREFIX_DEFAULT(.)

# configure automake
AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])
AM_SILENT_RULES([yes])
PAC_ADD_CPPFLAGS([-Wall -std=c11 -Wno-deprecated-declarations])

PAC_ADD_LIB_PATH(fabric,[rdma/fabric.h],[fi_getinfo],[ofi])
PAC_ADD_LIB_PATH(pmi,[pmi.h],[PMI_Init],[pmi])


PAC_ADD_ENABLE_MODE([asan],
                    [-g -pg -O0 -fsanitize=address],
                    [-fsanitize=address],
                    [enable address sanitizer])

# request libfabric-ofi
#kAC_LIB_HAVE_LINKFLAGS([fabric],[],[],[],AC_MSG_ERROR([enable to find libfabric]))
#AC_ARG_WITH([ofi],
#            [AS_HELP_STRING([--with-ofi@<:@=<path>@:>@],[where to find ofi])],
#            [],
#            [with_ofi=check])
#
##LIBOFI = AS_IF(



#dnl -- configs
#AC_CONFIG_SRCDIR(src)
#AC_CONFIG_AUX_DIR(config)
#
#dnl -- Check for toolchain and install components
#AC_PROG_INSTALL
#AC_PROG_LN_S
#AC_PROG_RANLIB
#AC_PROG_EGREP
#
#dnl -- check for standard c lib
#AC_C_CONST
#AC_C_INLINE
#AC_CHECK_INCLUDES_DEFAULT
#AC_CONFIG_SRCDIR(src/)
#AC_CONFIG_AUX_DIR(config)
#dnl - ask for a makefile
#AM_CONFIG_HEADER(config.h)

AC_OUTPUT
