/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */

#ifndef RMEM_H_
#define RMEM_H_

#include <stdatomic.h>
#include <stdint.h>
#include "rmem_utils.h"
#include "rmem_trigr.h"

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
#define m_countr_cas(a, e, d)                                                     \
    atomic_compare_exchange_strong_explicit(&(a)->val, e, d, m_default_mem_model, \
                                            m_default_mem_model)
#define m_countr_wcas(a, e, d)                                                     \
    atomic_compare_exchange_weak_explicit(&(a)->val, e, d, m_default_mem_model, \
                                            m_default_mem_model)

// list counter with acquire-release semantics
#define m_countr_acq_load(a)     atomic_load_explicit(&(a)->val, memory_order_acquire)
#define m_countr_rel_store(a, v) atomic_store_explicit(&(a)->val, v, memory_order_release)
#define m_countr_rel_fetch_add(a, v) atomic_fetch_add_explicit(&(a)->val, v, memory_order_release)
#define m_countr_rr_cas(a, e, d)                                                   \
    atomic_compare_exchange_strong_explicit(&(a)->val, e, d, memory_order_release, \
                                            memory_order_relaxed)

//--------------------------------------------------------------------------------------------------
typedef struct {
    atomic_intptr_t ptr;
} atomic_ptr_t;
#define m_atomicptr_init(a)     atomic_init(&(a)->ptr, 0)
#define m_atomicptr_load(a)     atomic_load_explicit(&(a)->ptr, m_default_mem_model)
#define m_atomicptr_copy(a, p)  atomic_store_explicit(&(a)->ptr, (p)->ptr, m_default_mem_model)
#define m_atomicptr_swap(a, p)  atomic_exchange_explicit(&(a)->ptr, p, m_default_mem_model)
#define m_atomicptr_store(a, p) atomic_store_explicit(&(a)->ptr, p, m_default_mem_model)

#define m_atomicptr_cas(a, e, p)                                     \
    atomic_compare_exchange_strong_explicit(&(a)->ptr, e, p, m_default_mem_model, \
                                            m_default_mem_model)
// #define m_atomicptr_cas_copy(a, e, p) m_atomicptr_cas(&(a)->ptr, e, (p)->ptr)

//--------------------------------------------------------------------------------------------------
// Multiple Producers, Single Consumer queue
typedef struct {
    rmem_trigr_ptr h_ready_ptr;  //!< ready variable
    rmem_trigr_ptr d_ready_ptr;  //!< device pointer to the ready variable
    atomic_ptr_t next;
} rmem_qnode_t;  // node
typedef struct {
    // progress count
    countr_t ongoing; // +1 when submited, -1 when executed
    // gpu triggered resources
    countr_t trigr_count;
    uint8_t* pool_bitmap;
    rmem_trigr_ptr h_trigr_pool;
    rmem_trigr_ptr d_trigr_pool;
    // queue navigation
    atomic_ptr_t head;
    atomic_ptr_t tail;
    atomic_ptr_t prev;
    atomic_ptr_t curnt;
} rmem_qmpsc_t;  // queue
void rmem_qmpsc_enq(rmem_qmpsc_t* q, rmem_qnode_t* elem);
void rmem_qmpsc_deq(rmem_qmpsc_t* q, rmem_qnode_t** elem);
void rmem_qmpsc_deq_ifready(rmem_qmpsc_t* q, rmem_qnode_t** elem);

#endif
