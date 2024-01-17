#ifndef RMEM_lmpsc_H_
#define RMEM_lmpsc_H_

#include <stdint.h>
#include <pthread.h>
#include "rmem.h"
#include "rmem_trigr.h"

#define m_rmem_lnode_kind_null  0x00
#define m_rmem_lnode_kind_rma   0x01
#define m_rmem_lnode_kind_wait  0x02
#define m_rmem_lnode_kind_compl 0x04

typedef enum {
    LNODE_KIND_NULL,
    LNODE_KIND_RMA,
    LNODE_KIND_COMPL,
} rmem_lnode_kind_t;

// Multiple Producers, Single Consumer queue - based on a single list
typedef struct {
    rmem_trigr_ptr h_ready_ptr;  //!< ready variable
    rmem_trigr_ptr d_ready_ptr;  //!< device pointer to the ready variable
    rmem_lnode_kind_t kind;      //!< kind of operation on the node
} rmem_lnode_t;                  // node
//
typedef struct {
    // progress count: number of operations not performed yet
    countr_t ongoing;  // +1 when submited, -1 when executed
    // gpu triggered resources
    countr_t list_count; //<! count the number of operations currently enqueued (done or not)
    uint8_t* list_bm;  // bitmap
    rmem_trigr_ptr h_trigr_list;
    rmem_trigr_ptr d_trigr_list;
    pthread_mutex_t reset;  // lock to reset the list
} rmem_lmpsc_t;             // queue

//
void rmem_lmpsc_create(rmem_lmpsc_t* q);
void rmem_lmpsc_destroy(rmem_lmpsc_t* q);
void rmem_lmpsc_reset(rmem_lmpsc_t* q);
void rmem_lmpsc_done(rmem_lmpsc_t* q, rmem_lnode_t* elem);

rmem_trigr_ptr rmem_lmpsc_enq(rmem_lmpsc_t* q, rmem_lnode_t* elem);
rmem_trigr_ptr rmem_lmpsc_enq_val(rmem_lmpsc_t* q, rmem_lnode_t* elem, bool value);

void rmem_lmpsc_deq_ifready(rmem_lmpsc_t* q, rmem_lnode_t** elem, int* idx, int* cnt);
void rmem_lmpsc_test_cancel(rmem_lmpsc_t* q, int* cnt);

#endif
