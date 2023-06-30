
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

//--------------------------------------------------------------------------------------------------
typedef struct {
    atomic_intptr_t ptr;
} atomic_ptr_t;
#define m_atomicptr_init(a)        atomic_init(&(a)->ptr, 0)
#define m_atomicptr_load(a)        atomic_load_explicit(&(a)->ptr, m_default_mem_model)
#define m_atomicptr_copy(a, p)     atomic_store_explicit(&(a)->ptr, (p)->ptr, m_default_mem_model)
#define m_atomicptr_store(a, p)    atomic_store_explicit(&(a)->ptr, p, m_default_mem_model)
#define m_atomicptr_exchange(a, p) atomic_exchange_explicit(&(a)->ptr, p, m_default_mem_model)

#define m_atomicptr_compare_exchange(a, e, p)                                     \
    atomic_compare_exchange_strong_explicit(&(a)->ptr, e, p, m_default_mem_model, \
                                            m_default_mem_model)
#define m_atomicptr_compare_copy(a, e, p)                                                \
    atomic_compare_exchange_strong_explicit(&(a)->ptr, e, (p)->ptr, m_default_mem_model, \
                                            m_default_mem_model)
//--------------------------------------------------------------------------------------------------
// Multiple Producers, Single Consumer queue
typedef struct {
    volatile int ready;
    atomic_ptr_t next;
} rmem_qnode_t;  // node
typedef struct {
    atomic_ptr_t head;
    atomic_ptr_t tail;
    atomic_ptr_t prev;
    atomic_ptr_t curnt;
} rmem_qmpsc_t;  // queue
void rmem_qmpsc_enq(rmem_qmpsc_t* q, rmem_qnode_t* elem);
void rmem_qmpsc_deq(rmem_qmpsc_t* q, rmem_qnode_t** elem);
void rmem_qmpsc_deq_ifready(rmem_qmpsc_t* q, rmem_qnode_t** elem);



#endif
