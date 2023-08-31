#ifndef RMEM_ARGP_H_
#define RMEM_ARGP_H_

#include <stdlib.h>
#include "ofi.h"
#include <argp.h>

#ifdef GIT_COMMIT
const char *argp_program_version ="rmem - "GIT_COMMIT;
#else
const char *argp_program_version ="rmem";
#endif


/* Program documentation. */
static char doc[] ="Remote MEMory - low latency communications\n -- (c) Argonne National Laboratory --";

/* A description of the arguments we accept. */
static char args_doc[] = "";

/* The options we understand. */
static struct argp_option options[] = {
    {"signal", 's', "MODE", 0, "signal mechanism: atomic or cq_data", 1},
    {"remote-complete", 'c', "MODE", 0,
     "remote completion mechanism: fence, cq_data, counter, or delivery", 1},
    {"ready-to-receive", 'r', "MODE", 0, "ready-to-receive mechanism: atomic, tag or am", 1},
    {"down-to-close", 'd', "MODE", 0, "down-to-close mechanism: tag or am", 1},
    {0}};

// /* Used by main to communicate with parse_opt. */
// struct argp_rmem {
//     int sig_mode;
//     int rtr_mode;
//     int rcmpl_mode;
// };

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    /* Get the input argument from argp_parse, which we
       know is a pointer to our arguments structure. */
    ofi_mode_t *arguments = state->input;
    arguments->sig_mode = M_OFI_SIG_NULL;
    arguments->rtr_mode = M_OFI_RTR_NULL;
    arguments->dtc_mode = M_OFI_DTC_NULL;
    arguments->rcmpl_mode = M_OFI_RCMPL_NULL;

    switch (key) {
        //------------------------------------------------------------------------------------------
        // signal
        case 's':
            if (0 == strcmp(arg, "atomic")) {
                arguments->sig_mode = M_OFI_SIG_ATOMIC;
            } else if (0 == strcmp(arg, "cq_data")) {
                arguments->sig_mode = M_OFI_SIG_CQ_DATA;
            } else {
                m_log("unknown value in signal argument: %s", arg);
                argp_usage(state);
            }
            break;
        //------------------------------------------------------------------------------------------
        // remote completion
        case 'c':
            m_log("arg = %s", arg);
            if (0 == strcmp(arg, "fence")) {
                arguments->rcmpl_mode = M_OFI_RCMPL_FENCE;
            } else if (0 == strcmp(arg, "cq_data")) {
                arguments->rcmpl_mode = M_OFI_RCMPL_CQ_DATA;
            } else if (0 == strcmp(arg, "counter")) {
                arguments->rcmpl_mode = M_OFI_RCMPL_REMOTE_CNTR;
            } else if (0 == strcmp(arg, "delivery")) {
                arguments->rcmpl_mode = M_OFI_RCMPL_DELIV_COMPL;
            } else {
                m_log("unknown value in remote completion argument: %s", arg);
                argp_usage(state);
            }
            break;
        //------------------------------------------------------------------------------------------
        // ready to receive
        case 'r':
            if (0 == strcmp(arg, "tag")) {
                arguments->rtr_mode = M_OFI_RTR_TAGGED;
            } else if (0 == strcmp(arg, "atomic")) {
                arguments->rtr_mode = M_OFI_RTR_ATOMIC;
            } else if (0 == strcmp(arg, "am")) {
                arguments->rtr_mode = M_OFI_RTR_MSG;
            } else {
                m_log("unknown value in ready-to-receive argument: %s", arg);
                argp_usage(state);
            }
            break;
        //------------------------------------------------------------------------------------------
        // down to close
        case 'd':
            if (0 == strcmp(arg, "am")) {
                arguments->dtc_mode = M_OFI_DTC_MSG;
            } else if (0 == strcmp(arg, "tag")) {
                arguments->dtc_mode = M_OFI_DTC_TAGGED;
            } else {
                m_log("unknown value in down-to-close argument: %s", arg);
                argp_usage(state);
            }
            break;
        //------------------------------------------------------------------------------------------
        // something else
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

/* Our argp parser. */
static struct argp argp = {options, parse_opt, 0, doc};

#endif
