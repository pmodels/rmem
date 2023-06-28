/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "rmem_utils.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief prints the backtrace history
 *
 * based on
 * - https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
 * - https://gist.github.com/fmela/591333/c64f4eb86037bb237862a8283df70cdfc25f01d3
 * - https://linux.die.net/man/3/dladdr
 *
 * The symbols generated might be weird if the demangler doesn't work correctly
 * if you don't see any function name, add the option `-rdynamic` to the linker's flag
 *
 */
void PrintBackTrace() {
    //--------------------------------------------------------------------------
#if (1 == M_BACKTRACE)
    // get the addresses of the function currently in execution
    void *call_stack[M_BACKTRACE_HISTORY];  // array of
    int size = backtrace(call_stack, M_BACKTRACE_HISTORY);

    // transform the pointers into readable symbols
    char **strings;
    strings = backtrace_symbols(call_stack, size);
    if (strings != NULL) {
        m_log("--------------------- CALL STACK ----------------------");
        // we start at 1 to not display this function
        for (int i = 1; i < size; i++) {
            //  display the demangled name if we succeeded, weird name if not
            m_log("%s", strings[i]);
        }
        m_log("-------------------------------------------------------");
    }
    // free the different string with the names
    free(strings);
#endif
    //--------------------------------------------------------------------------
}
