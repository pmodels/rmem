/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
#ifndef PMI_UTILS_H_
#define PMI_UTILS_H_

#include <pmi.h>
#include <stddef.h>

#ifndef NDEBUG
#define m_pmi_call(func)                                                          \
    do {                                                                          \
        int m_pmi_call_res = func;                                                \
        m_assert(m_pmi_call_res == PMI_SUCCESS, "PMI ERROR: %d", m_pmi_call_res); \
    } while (0)
#else
#define m_pmi_call(func) \
    do {                 \
        func;            \
    } while (0)
#endif

int pmi_init();
int pmi_get_comm_id(int* id_world, int* n_world);
int pmi_allgather(const size_t addr_len, const void* addr, void** addr_world);
int pmi_finalize();

#endif
