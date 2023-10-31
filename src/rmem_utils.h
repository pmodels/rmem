/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef RMEM_UTILS_H_
#define RMEM_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pmi.h>


/*
 * note: using do{}while(0) is the best way to create multi-line macros
 * */

#ifdef NO_BTRACE
#define M_BACKTRACE 0
#else
#define M_BACKTRACE 1
#endif

#define m_success EXIT_SUCCESS
#define m_failure EXIT_FAILURE

#define M_BACKTRACE_HISTORY 50

// alignment on the cache line size
#define m_alignment 64

//==============================================================================
void PrintBackTrace();

//==============================================================================
#define m_max(a, b)                                      \
    ({                                                   \
        __typeof__(a) m_max_a_ = (a);                    \
        __typeof__(b) m_max_b_ = (b);                    \
        (m_max_a_ > m_max_b_) ? (m_max_a_) : (m_max_b_); \
    })

//------------------------------------------------------------------------------
#define m_min(a, b)                                      \
    ({                                                   \
        __typeof__(a) m_min_a_ = (a);                    \
        __typeof__(b) m_min_b_ = (b);                    \
        (m_min_a_ < m_min_b_) ? (m_min_a_) : (m_min_b_); \
    })

//------------------------------------------------------------------------------
#define m_sign(a)                                                \
    ({                                                           \
        __typeof__(a) m_sign_a_ = (a);                           \
        __typeof__(a) m_sign_zero_ = 0;                          \
        (m_sign_zero_ < m_sign_a_) - (m_sign_a_ < m_sign_zero_); \
    })

#define m_malloc(s)                                                           \
    ({                                                                        \
        size_t m_malloc_size = ((size_t)s + (m_alignment - 1)) / m_alignment; \
        aligned_alloc(m_alignment, m_malloc_size* m_alignment);               \
    })

//==============================================================================
// LOGGING
//==============================================================================
//------------------------------------------------------------------------------
/**
 * @brief m_log will be displayed as a log, either by every rank or only by the master (given
 * LOG_ALLRANKS)
 *
 */
#ifndef LOG_MUTE
#define m_log_def(header_name, format, ...)                               \
    do {                                                                  \
        fprintf(stdout, "[" header_name "] " format "\n", ##__VA_ARGS__); \
        fflush(stdout);                                                   \
    } while (0)
#else  // LOG_MUTE
#define m_log_def(header_name, format, ...) \
    { ((void)0); }
#endif  // LOG_MUTE

#define m_log(format, ...) m_log_def("rmem", format, ##__VA_ARGS__)

//------------------------------------------------------------------------------
/**
 * @brief m_verb will be displayed if VERBOSE is enabled
 *
 */
#ifdef VERBOSE
#define m_verb_def(header_name, format, ...)                              \
    do {                                                                  \
        fprintf(stdout, "[" header_name "] " format "\n", ##__VA_ARGS__); \
    } while (0)
#else  // VERBOSE
#define m_verb_def(header_name, format, ...) \
    { ((void)0); }
#endif  // VERBOSE

#define m_verb(format, ...) m_verb_def("rmem", format, ##__VA_ARGS__)

//------------------------------------------------------------------------------
/**
 * @brief m_assert defines the assertion call, disabled if NDEBUG is asked
 *
 */
#ifdef NDEBUG
#define m_assert_def(cond, ...) \
    { ((void)0); }
#else
#define m_assert_def(header_name, cond, ...)                                                \
    do {                                                                                    \
        bool m_assert_def_cond_ = (bool)(cond);                                             \
        if (!(m_assert_def_cond_)) {                                                        \
            char m_assert_def_msg_[1024];                                                   \
            sprintf(m_assert_def_msg_, __VA_ARGS__);                                        \
            fprintf(stdout, "[%s-assert] '%s' FAILED: %s (at %s:%d)\n", header_name, #cond, \
                    m_assert_def_msg_, __FILE__, __LINE__);                                 \
            PrintBackTrace();                                                               \
            fflush(stdout);                                                                 \
            PMI_Abort(EXIT_FAILURE, NULL);                                                  \
        }                                                                                   \
    } while (0)
#endif

#define m_assert(cond, ...) m_assert_def("rmem", cond, ##__VA_ARGS__)

//------------------------------------------------------------------------------
#define m_error_def(header_name, format, ...)                             \
    do {                                                                  \
        fprintf(stderr, "[" header_name "] " format "\n", ##__VA_ARGS__); \
        fflush(stderr);                                                   \
        return m_failure;                                                 \
    } while (0)

#define m_error(cond, ...) m_error_def("rmem", cond, ##__VA_ARGS__)

//------------------------------------------------------------------------------
#ifndef NDEBUG
#define m_rmem_call(func)                                                       \
    do {                                                                        \
        int m_pmi_call_res = func;                                              \
        m_assert(m_pmi_call_res == m_success, "PMI ERROR: %d", m_pmi_call_res); \
    } while (0)
#else
#define m_rmem_call(func) \
    do {                  \
        func;             \
    } while (0)
#endif

#endif
