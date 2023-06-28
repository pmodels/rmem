
#ifndef RMEM_H_
#define RMEM_H_

#include <stdatomic.h>
#include <stdint.h>
#include "rmem_utils.h"
//--------------------------------------------------------------------------------------------------
#define m_default_mem_model memory_order_relaxed
typedef struct {
    atomic_int val;
} countr_t;
#define m_countr_init(a)         atomic_init(&(a)->val, 0)
#define m_countr_load(a)         atomic_load_explicit(&(a)->val, m_default_mem_model)
#define m_countr_store(a, v)     atomic_store_explicit(&(a)->val, v, m_default_mem_model)
#define m_countr_exchange(a, v)  atomic_exchange_explicit(&(a)->val, v, m_default_mem_model)
#define m_countr_fetch_add(a, v) atomic_fetch_add_explicit(&(a)->val, v, m_default_mem_model)


#endif
