/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef OFI_H_
#define OFI_H_
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include "rdma/fi_atomic.h"

//--------------------------------------------------------------------------------------------------
#define OFI_CQ_FORMAT             FI_CQ_FORMAT_CONTEXT
typedef struct fi_cq_entry ofi_cq_entry;

//--------------------------------------------------------------------------------------------------
#ifndef NDEBUG
#define m_ofi_call(func)                                                              \
    do {                                                                              \
        int m_ofi_call_res = func;                                                    \
        m_assert(m_ofi_call_res >= 0, "OFI ERROR: %s", fi_strerror(-m_ofi_call_res)); \
    } while (0)
#else
#define m_ofi_call(func) \
    do {                 \
        func;            \
    } while (0)
#endif

//--------------------------------------------------------------------------------------------------
// TAGGED SEND-RECV
#define m_ofi_tag_tot    48  // total number of bits in the tag
#define m_ofi_tag_ctx    7   // number of bits to encode the comm context id
#define m_ofi_tag_intern 5
#define m_ofi_tag_avail  (m_ofi_tag_tot - m_ofi_tag_ctx - m_ofi_tag_intern)

#define m_ofi_tag_ctx_shift    (m_ofi_tag_avail)
#define m_ofi_tag_ctx_bits     ((uint64_t)0x7f)  // mask to get the context
#define m_ofi_tag_ctx_mask     ((m_ofi_tag_ctx_bits) << m_ofi_tag_ctx_shift)
#define m_ofi_tag_get_ctx(tag) ((tag & m_ofi_tag_ctx_mask) >> m_ofi_tag_ctx_shift)
#define m_ofi_tag_set_ctx(ctx) ((ctx << m_ofi_tag_ctx_shift) & m_ofi_tag_ctx_mask)

inline uint64_t ofi_set_tag(const int ctx_id, const int tag) {
    uint64_t ofi_tag = ctx_id;
    return ((ofi_tag << m_ofi_tag_ctx_shift) & m_ofi_tag_ctx_mask) | tag;
}

//--------------------------------------------------------------------------------------------------
// rma flags sent either as REMOTE_CQ_DATA or as 64 bit msg
// 64 bits total: [ 5b header | 1b post | 1b complete | ... | 32 bits # of operations]
#define m_ofi_data_bit_post (m_ofi_tag_tot - 1)
#define m_ofi_data_bit_cmpl (m_ofi_tag_tot - 2)
#define m_ofi_tag_bit_sync  (m_ofi_tag_tot - 3)

#define m_ofi_data_set_post    ((uint64_t)0x1 << m_ofi_data_bit_post)
#define m_ofi_data_set_cmpl    ((uint64_t)0x1 << m_ofi_data_bit_cmpl)
#define m_ofi_data_get_post(a) ((a >> m_ofi_data_bit_post) & 0x1)
#define m_ofi_data_get_cmpl(a) ((a >> m_ofi_data_bit_cmpl) & 0x1)
#define m_ofi_data_set_nops(a) (((uint64_t)a) & ((uint64_t)0xffffffff))
#define m_ofi_data_get_nops(a) ((uint32_t)(a & 0xffffffff))
#define m_ofi_tag_sync         ((uint64_t)0x1 << m_ofi_tag_bit_sync)

//--------------------------------------------------------------------------------------------------
#define m_ofi_cq_kind_sync (0x01)  // 0000 0001
#define m_ofi_cq_kind_rqst (0x02)  // 0000 0010

//--------------------------------------------------------------------------------------------------
typedef struct {
    atomic_int val;
} countr_t;
//--------------------------------------------------------------------------------------------------
#define m_countr_init(a)         atomic_init(&(a)->val, 0)
#define m_countr_load(a)         atomic_load_explicit(&(a)->val, memory_order_relaxed)
#define m_countr_store(a, v)     atomic_store_explicit(&(a)->val, v, memory_order_relaxed)
#define m_countr_exchange(a, v)  atomic_exchange_explicit(&(a)->val, v, memory_order_relaxed)
#define m_countr_fetch_add(a, v) atomic_fetch_add_explicit(&(a)->val, v, memory_order_relaxed)


//--------------------------------------------------------------------------------------------------
// communication context
typedef struct {
    // shared rx/tx contexts
    struct fid_stx* stx;
    struct fid_ep* srx;

    // point to point resources
    struct fid_ep* p2p_ep;
    struct fid_cq* p2p_cq;
    struct fid_av* p2p_av;

    fi_addr_t* p2p_addr;
} ofi_ctx_t;

// communicator
typedef struct {
    int size;
    int rank;

    struct fi_info* prov;
    struct fid_fabric* fabric;
    struct fid_domain* domain;

    int n_ctx;
    ofi_ctx_t* ctx;
} ofi_comm_t;

/**
 * @brief data-structure used in the cq when an operation has completed
 */
typedef struct {
    //  ofi structures link to the fi_cq
    struct fid_cq* cq;
    // context for the CQ entry
    struct fi_context ctx;

    // kind parameter
    uint8_t kind;
    union {
        struct {
            countr_t* flag;
        } rqst;  // kind == m_ofi_cq_kind_rqst
        struct {
            countr_t* cntr;
            uint64_t buf;    // buffer for communication
        } sync;  // kind == m_ofi_cq_kind_sync
    };
} ofi_cqdata_t;

typedef struct {
    // user provided information
    void* buf;     // address of the buffer
    size_t count;  // count in bytes
    int peer;      // destination/origin rank
    int tag;       // user tag

    // implementation specifics
    struct {
        countr_t completed;
        // completion queue data
        ofi_cqdata_t cq;
        // data description and ofi msg
        struct iovec iov;
        struct fi_msg_tagged msg;
    } ofi;
} ofi_p2p_t;

typedef struct {
    struct fid_ep* ep;
    struct fid_ep* srx;
    struct fid_cq* cq;       // completion queue for RECEIVE and REMOTE_DATA
    struct fid_av* av;       // address vector
    struct fid_cntr* rcntr;  // Remotely issued CouNTeR put and get
    struct fid_cntr* ccntr;  // Completed CouNTeR put and get
    fi_addr_t* addr;         // address list
} ofi_rma_trx_t;

typedef struct {
    // signal counter
    uint32_t inc; // increment value, always 1
    uint32_t val; // actual counter value
    // structs for fi_atomics
    uint64_t* key_list;  // list of remote keys
    struct fid_mr* mr;
} ofi_rma_sig_t;

typedef struct {
    countr_t epoch[3];
    countr_t* icntr;  // fi_write only!
    ofi_cqdata_t* cqdata;
} ofi_rma_sync_t;

typedef struct {
    // user provided information
    void* buf;     // address of the buffer
    size_t count;  // count in bytes
    int peer;      // destination/origin rank
    ssize_t disp;  // displacement compared to the rmem base ptr (in bytes)

    // implementation specifics
    struct {
        countr_t completed;
        // data description and ofi msg
        struct {
            uint64_t flags;
            struct iovec iov;
            struct fi_rma_iov riov;
            ofi_cqdata_t cq;
        } msg;
        struct {
            uint64_t flags;
            struct fi_ioc iov;
            struct fi_rma_ioc riov;
            struct fi_context ctx; // to replace by cqdata_t if RPUT_SIG is desired
        } sig;
        fi_addr_t addr;
        struct fid_ep* ep;
    } ofi;
} ofi_rma_t;

//-------------------------------------------------------------------------------------------------
// memory exposed to the world - public memory
typedef struct {
    // user defined buffer
    void* buf;
    size_t count;

    // ofi specifics
    struct {
        int n_rx;  // number of received context
        int n_tx;  // number of transmit contexts
        // buffer addresses
        struct fid_mr* mr;
        uint64_t* key_list;  // list of remote keys
        // transmit and receive contexts
        ofi_rma_trx_t* sync_trx;
        ofi_rma_trx_t* data_trx;
        // signaling
        ofi_rma_sig_t signal;
        ofi_rma_sync_t sync;
    } ofi;
} ofi_rmem_t;

// init and finalize - ofi_init.c
int ofi_init(ofi_comm_t* ofi);
int ofi_finalize(ofi_comm_t* ofi);

// utility function
inline int ofi_get_rank(ofi_comm_t* ofi) { return ofi->rank; }
inline int ofi_get_size(ofi_comm_t* ofi) { return ofi->size; }

//-------------------------------------------------------------------------------------------------
// create a point to point communication
int ofi_p2p_create(ofi_p2p_t* p2p, ofi_comm_t* comm);
int ofi_p2p_free(ofi_p2p_t* p2p);

// send/recv
int ofi_send_enqueue(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm);
int ofi_recv_enqueue(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm);

// progress
int ofi_progress(struct fid_cq* cq);
int ofi_p2p_wait(ofi_p2p_t* p2p);
int ofi_rma_wait(ofi_rma_t* p2p);

//-------------------------------------------------------------------------------------------------
// Remote memory management
int ofi_rmem_init(ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_free(ofi_rmem_t* mem, ofi_comm_t* comm);

int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);

// operation management
int ofi_rma_start(ofi_rma_t* rma);
int ofi_rma_free(ofi_rma_t* rma);

int ofi_put_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
int ofi_put_signal_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
int ofi_rput_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);

#endif
