dnl 
dnl Copyright (C) by Argonne National Laboratory
dnl  See COPYRIGHT in top-level directory
dnl 
dnl ==================================================================================================
dnl PAC_CHECK_HEADER_LIB(header.h, libname, function, action-if-yes, action-if-no)
dnl This macro checks for a header and lib.
dnl from https://github.com/pmodels/mpich/blob/main/confdb/aclocal_libs.m4
AC_DEFUN([PAC_CHECK_HEADER_LIB],[
    failure=no
    AC_CHECK_HEADER([$1],,failure=yes)
    AC_CHECK_LIB($2,$3,,failure=yes)
    if test "$failure" = "no" ; then
    $4
    else
    $5
    fi
])

dnl ==================================================================================================
dnl PAC_APPEND_FLAG([argument],[var])
dnl appends the given argument to the specified shell variable unless the argument is already present in the variable
dnl from https://github.com/pmodels/mpich/blob/main/confdb/aclocal_util.m4
  AC_DEFUN([PAC_APPEND_FLAG],[
	  AC_REQUIRE([AC_PROG_FGREP])
	  AS_IF(
		[echo "$$2" | $FGREP -e '$1' >/dev/null 2>&1],
		[echo "$2(='$$2') contains '$1', not appending" >&AS_MESSAGE_LOG_FD],
		[echo "$2(='$$2') does not contain '$1', appending" >&AS_MESSAGE_LOG_FD
		$2="$$2 $1"]
	    )
  ])

dnl ==================================================================================================
dnl PAC_ADD_LIB_PATH(name, header.h, function, abbrv)
dnl creates the option "--with-abbr" to load a library
dnl add the include and lib director of lib<name>.
dnl uses "header.h" to verify the existence of the header files and "function" to test linking to it
AC_DEFUN([PAC_ADD_LIB_PATH],[
    AC_MSG_NOTICE([>> searching for <$4>])
    dnl create the --with argument
    AC_ARG_WITH([$4],
        [AS_HELP_STRING([--with-$4=<path>],[specify path where <$4/lib$1> include directory and lib directory can be found])],
        [AS_CASE(["$withval"],
            [yes|no|''],
            [AC_MSG_WARN([--with@<:@out@:>@-$4=<path> expects a <path>, using system defaults])
            with_$4=""])],
        [with_$4=""])
    dnl add the correct path to the compilation flags
    AS_IF([test -n "${with_$4}"],
        [AS_IF( [test -d ${with_$4}],
            [
            AC_MSG_NOTICE([$4 path: ${with_$4}])
            PAC_APPEND_FLAG([-I${with_$4}/include],[CPPFLAGS])
            AS_IF( [test -d ${with_$4}/lib],
                [PAC_APPEND_FLAG([-L${with_$4}/lib -Wl,-rpath,${with_$4}/lib],[LDFLAGS])],)
            AS_IF( [test -d ${with_$4}/lib64],
                [PAC_APPEND_FLAG([-L${with_$4}/lib64 -Wl,-rpath,${with_$4}/lib64],[LDFLAGS])],)
            ],
            [AC_MSG_ERROR([path given for $4 is not valid])])],
        [AC_MSG_NOTICE([$1 path: system defaults])])
    dnl make sure we can link to the lib now
    PAC_CHECK_HEADER_LIB([$2],[$1],[$3],
        [AC_MSG_NOTICE([successfully linked to <$4/lib$1>]) ],
        [AC_MSG_ERROR([impossible to link correctly to <$4/lib$1>])])
])

dnl ==================================================================================================
dnl add $1 to CPPFLAGS and $2 to LDFLAGS + attempt to compile an empty program with it
AC_DEFUN([PAC_ADD_FLAGS],[
    AC_MSG_NOTICE([trying to add cc flag <$1> and ld flag <$2>])
    failure=no
    for flag in $1; do
        PAC_APPEND_FLAG([$1],[CPPFLAGS])
    done
    for flag in $2; do
        PAC_APPEND_FLAG([$2],[LDFLAGS])
    done

    dnl try to link a stupid program with it
    AC_LINK_IFELSE([AC_LANG_SOURCE([int main() { return 0; }])],,failure=yes)

    if test "$failure" = "yes"; then
        AC_MSG_ERROR([impossible to compile with cc flag <$1> and ld flags <$2>])
    else
        AC_MSG_NOTICE([successfully compiled with cc flag <$1> and ld flag <$2>])
    fi
])

dnl ==================================================================================================
dnl - the name of the mode, will be '--enable-name'
dnl - the CPPFLAGS to add
dnl - the LDFLAGS to add
dnl - a description
AC_DEFUN([PAC_ADD_ENABLE_MODE],[
    AC_ARG_ENABLE([$1],
        [AS_HELP_STRING([--enable-$1],[$4])],
        [PAC_ADD_FLAGS([$2],[$3])],[])
])

dnl ==================================================================================================
dnl add a list of CPPLFAGS
AC_DEFUN([PAC_ADD_CPPFLAGS],[
    PAC_ADD_FLAGS([$1],[ ])
])


