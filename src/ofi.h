/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef OFI_H_
#define OFI_H_
#include <assert.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "rdma/fi_atomic.h"
#include "rmem.h"


#define M_SYNC_ATOMIC 1

#ifndef NO_RMA_EVENT
#define M_SYNC_RMA_EVENT 1
#else
#define M_SYNC_RMA_EVENT 0
#endif

#ifndef NO_WRITE_DATA
#define M_WRITE_DATA 1
#else
#define M_WRITE_DATA 0
#endif

static_assert(!(!M_WRITE_DATA && !M_SYNC_RMA_EVENT),
              "no RMA events and no write data is not supported. Pls review compilation flags "
              "`-DNO_RMA_EVENT` and `-DNO_WRITE_DATA`");

#if (!M_SYNC_RMA_EVENT || M_WRITE_DATA)
#define OFI_CQ_FORMAT FI_CQ_FORMAT_DATA
typedef struct fi_cq_data_entry ofi_cq_entry;
#else
#define OFI_CQ_FORMAT FI_CQ_FORMAT_CONTEXT
typedef struct fi_cq_entry ofi_cq_entry;
#endif

//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
#define m_ofi_call_again(func, progress)                                                  \
    do {                                                                                  \
        int m_ofi_call_res = func;                                                        \
        if (m_ofi_call_res == -FI_EAGAIN) {                                               \
            /* if the error code is FI_EAGAIN, try to progress and do it again*/          \
            ofi_progress(progress);                                                       \
        } else {                                                                          \
            /* if it's a success, leave*/                                                 \
            m_assert(m_ofi_call_res >= 0, "OFI ERROR: %s", fi_strerror(-m_ofi_call_res)); \
            break;                                                                        \
        }                                                                                 \
    } while (1)

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


#ifndef NDEBUG
#define m_pthread_call(func)                                                        \
    do {                                                                            \
        int m_pthread_call_res = func;                                              \
        m_assert(m_pthread_call_res == 0, "PTHREAD ERROR: %d", m_pthread_call_res); \
    } while (0)
#else
#define m_pthread_call(func) \
    do {                     \
        func;                \
    } while (0)
#endif

//--------------------------------------------------------------------------------------------------
// TAGGED SEND-RECV
#define m_ofi_tag_tot    48  // total number of bits in the tag
#define m_ofi_tag_ctx    7   // number of bits to encode the comm context id
#define m_ofi_tag_intern 9
#define m_ofi_tag_usr    32
#define m_ofi_tag_avail  (m_ofi_tag_tot - m_ofi_tag_intern - m_ofi_tag_ctx )

// internal
#define m_ofi_tag_bit_ps (m_ofi_tag_tot - 1)
#define m_ofi_tag_bit_cw (m_ofi_tag_tot - 2)
#define m_ofi_tag_set_ps ((uint64_t)0x1 << m_ofi_tag_bit_ps)
#define m_ofi_tag_set_cw ((uint64_t)0x1 << m_ofi_tag_bit_cw)
// ctx
#define m_ofi_tag_ctx_shift    (m_ofi_tag_avail - m_ofi_tag_intern)
#define m_ofi_tag_ctx_bits     ((uint64_t)0x7f)  // mask to get the context
#define m_ofi_tag_ctx_mask     ((m_ofi_tag_ctx_bits) << m_ofi_tag_ctx_shift)
#define m_ofi_tag_get_ctx(tag) ((tag & m_ofi_tag_ctx_mask) >> m_ofi_tag_ctx_shift)
#define m_ofi_tag_set_ctx(ctx) ((ctx << m_ofi_tag_ctx_shift) & m_ofi_tag_ctx_mask)
//usr
#define m_ofi_tag_set_usr(a) (((uint64_t)a) & ((uint64_t)0xffffffff))

static inline uint64_t ofi_set_tag(const int ctx_id, const int tag) {
    m_assert(m_ofi_tag_usr <= m_ofi_tag_avail,
             "the number of available bit for the tag is too short: %d vs %d", m_ofi_tag_usr,
             m_ofi_tag_avail);
    uint64_t ofi_tag = ctx_id;
    return ((ofi_tag << m_ofi_tag_ctx_shift) & m_ofi_tag_ctx_mask) | m_ofi_tag_set_usr(tag);
}

//--------------------------------------------------------------------------------------------------
// rma flags sent either as REMOTE_CQ_DATA or as 64 bit msg
// all the providers are required to support 4bytes, so we use that value
// 32 bits total: [ unsu | 1b post | 1b complete | ... | 16 bits # of operations]
#define m_ofi_data_tot    32
#define m_ofi_data_intern 16
#define m_ofi_data_avail  (m_ofi_data_tot - m_ofi_data_intern)

#define m_ofi_data_bit_post (m_ofi_data_tot - 1)
#define m_ofi_data_bit_cmpl (m_ofi_data_tot - 2)
#define m_ofi_data_bit_rcq  (m_ofi_data_tot - 3)
#define m_ofi_data_bit_sig  (m_ofi_data_tot - 4)

#define m_ofi_data_set_rcq     ((uint64_t)0x1 << m_ofi_data_bit_rcq)
#define m_ofi_data_set_sig     ((uint64_t)0x1 << m_ofi_data_bit_sig)
#define m_ofi_data_set_post    ((uint64_t)0x1 << m_ofi_data_bit_post)
#define m_ofi_data_set_cmpl    ((uint64_t)0x1 << m_ofi_data_bit_cmpl)
#define m_ofi_data_get_rcq(a)  ((a >> m_ofi_data_bit_rcq) & 0x1)
#define m_ofi_data_get_sig(a)  ((a >> m_ofi_data_bit_sig) & 0x1)
#define m_ofi_data_get_post(a) ((a >> m_ofi_data_bit_post) & 0x1)
#define m_ofi_data_get_cmpl(a) ((a >> m_ofi_data_bit_cmpl) & 0x1)
#define m_ofi_data_mask_nops (((uint64_t)0x1 << m_ofi_data_avail) -1 )
#define m_ofi_data_set_nops(a) (((uint64_t)a) & m_ofi_data_mask_nops)
#define m_ofi_data_get_nops(a) ((uint32_t)(a & m_ofi_data_mask_nops))

//--------------------------------------------------------------------------------------------------
// if true need to bump the local counter
#define m_ofi_cq_inc_local (0x80)  // 1000 0000
// define the opeation for after
#define m_ofi_cq_kind_null (0x01)  // 0000 0001
#define m_ofi_cq_kind_sync (0x02)  // 0000 0010
#define m_ofi_cq_kind_rqst (0x04)  // 0000 0100

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

    uint64_t unique_mr_key; // used for MR
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
    // context mandatory the CQ entry
    struct fi_context ctx;
    // kind parameter
    uint8_t kind;
    union {
        struct {
            countr_t busy;  // completed if 0
        } rqst;             // kind == m_ofi_cq_kind_rqst
        struct { 
            // array of epochs
            countr_t* cntr;
            // communication buffer
            uint64_t buf;  
            void* buf_desc; 
            struct fid_mr* buf_mr;
        } sync;              // kind == m_ofi_cq_kind_sync
    };
} ofi_cqdata_t;

typedef struct {
    //  ofi structures link to the fi_cq
    struct fid_cq* cq;
    // fallback context pointer, used if the ctx received is NULL, aka the entry is a REMOTE_CQ_DATA
    // value cannot be null if used explicitely to progress 
    void* fallback_ctx;
} ofi_progress_t;

typedef enum {
    P2P_OPT_SEND,
    P2P_OPT_RECV,
} p2p_opt_t;
typedef struct {
    // user provided information
    void* buf;     // address of the buffer
    size_t count;  // count in bytes
    int peer;      // destination/origin rank
    int tag;       // user tag

    // implementation specifics
    struct {
        ofi_progress_t progress;  // to make progress
        // completion queue data
        ofi_cqdata_t cq;
        // mr for MR_LOCAL
        void* desc_local;
        struct fid_mr* mr_local;
        // iovs
        struct iovec iov;
        struct fi_msg_tagged msg;
        // endpoint
        p2p_opt_t kind;
        uint64_t flags;
        struct fid_ep* ep;
    } ofi;
} ofi_p2p_t;

typedef struct {
    fi_addr_t* addr;  // address list
    struct fid_ep* ep;
    struct fid_ep* srx;
    struct fid_cq* cq;       // completion queue for RECEIVE and REMOTE_DATA
    struct fid_av* av;       // address vector
} ofi_rma_trx_t;

#if (!M_WRITE_DATA)
typedef struct {
    // signal counter - must be allocated to comply with FI_MR_ALLOCATE
    uint32_t* inc;  // increment value, always 1
    uint32_t* val;  // actual counter value
    // mr for MR_LOCAL
    void* desc_local;
    struct fid_mr* mr_local;
    // structs for fi_atomics
    uint64_t* base_list;  // list of base addresses
    uint64_t* key_list;   // list of remote keys
    struct fid_mr* mr;
    // remote counters for the signal, always available per the static assert above
    struct fid_cntr* scntr;
} ofi_rma_sig_t;
#endif

#define m_rma_epoch_post(e)    (e + 0)                 // posted
#define m_rma_epoch_cmpl(e)    (e + 1)                 // completed
#define m_rma_epoch_remote(e)  (e + 2)                 // remote fi_write
#define m_rma_epoch_local(e)   (e + 3)                 // local (tsend, signal, fi_write/read)
#define m_rma_mepoch_post(m)   (m->ofi.sync.epch + 0)  // posted
#define m_rma_mepoch_cmpl(m)   (m->ofi.sync.epch + 1)  // completed
#define m_rma_mepoch_remote(m) (m->ofi.sync.epch + 2)  // remote fi_write
#define m_rma_mepoch_local(m)  (m->ofi.sync.epch + 3)  // local (tsend, signal, fi_write/read)

#if (M_WRITE_DATA)
#define m_rma_epoch_signal(e)  (e + 4)                 // remote signal
#define m_rma_mepoch_signal(m) (m->ofi.sync.epch + 4)  // remote signal
#define m_rma_n_epoch          5
#else
#define m_rma_n_epoch 4
#endif

typedef struct {
#if (M_SYNC_ATOMIC)
    struct {
        uint32_t* inc;  // increment value, always 1
        uint32_t* res;  // result, used to read val
        uint32_t* val;  // actual value
        // mr for MR_LOCAL for INC
        void* desc_local_inc;
        struct fid_mr* mr_local_inc;
        // mr for MR_LOCAL for RES
        void* desc_local_res;
        struct fid_mr* mr_local_res;
        // structs for fi_atomics for VAL
        uint64_t* base_list;  // list of base addresses
        uint64_t* key_list;   // list of remote keys
        void* desc;
        struct fid_mr* mr;
    } post;
#endif
    countr_t isig;  //  issued signal (not sent to the target but still need to wait for completion)
    countr_t epch[m_rma_n_epoch];
    countr_t* icntr;       // array of fi_write counter (for each rank)
    ofi_cqdata_t* cqdata_ps;  // completion data for each rank post-start
    ofi_cqdata_t* cqdata_cw;  // completion data for each rank complete-wait
} ofi_rma_sync_t;

typedef struct {
    // user provided information
    void* buf;     // address of the buffer
    size_t count;  // count in bytes
    int peer;      // destination/origin rank
    ssize_t disp;  // displacement compared to the rmem base ptr (in bytes)

    // implementation specifics
    struct {
        ofi_progress_t progress;  // used to trigger progress
        // data description and ofi msg
        struct {
            uint64_t flags;
            // mr for MR_LOCAL for the local buffer associated to the operation
            void* desc_local;
            struct fid_mr* mr_local;
            // iovs
            struct iovec iov;
            struct fi_rma_iov riov;
            ofi_cqdata_t cq;
        } msg;
        struct {
#if (M_WRITE_DATA)
            uint64_t data;
#else
            uint64_t flags;
            // iovs
            struct fi_ioc iov;
            struct fi_rma_ioc riov;
            ofi_cqdata_t cq;
            // struct fi_context ctx;  // to replace by cqdata_t if RPUT_SIG is desired
#endif
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
        fi_addr_t* base_list;  // list of base addresses
        uint64_t* key_list;  // list of remote keys
        // transmit and receive contexts
        ofi_rma_trx_t* sync_trx;
        ofi_rma_trx_t* data_trx;
        // completion and remote counter global for all trx
#if (M_SYNC_RMA_EVENT)
        struct fid_cntr* rcntr;  // Completed CouNTeR put and get
#endif
            // signaling
#if (!M_WRITE_DATA)
        ofi_rma_sig_t signal;
#endif
        // synchronization (PSCW)
        ofi_rma_sync_t sync;
    } ofi;
} ofi_rmem_t;

// init and finalize - ofi_init.c
int ofi_init(ofi_comm_t* ofi);
int ofi_finalize(ofi_comm_t* ofi);

static inline char* ofi_name(ofi_comm_t* ofi) { return ofi->prov->fabric_attr->prov_name; }

// utility function
static inline int ofi_get_rank(ofi_comm_t* ofi) { return ofi->rank; }
static inline int ofi_get_size(ofi_comm_t* ofi) { return ofi->size; }

//-------------------------------------------------------------------------------------------------
// send/recv
int ofi_send_init(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm);
int ofi_recv_init(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm);
int ofi_p2p_start(ofi_p2p_t* p2p);
int ofi_p2p_free(ofi_p2p_t* p2p);

// progress
int ofi_progress(ofi_progress_t* progress);
int ofi_p2p_wait(ofi_p2p_t* p2p);

//-------------------------------------------------------------------------------------------------
// Remote memory management
int ofi_rmem_init(ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_free(ofi_rmem_t* mem, ofi_comm_t* comm);

// PSCW 101
int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
// fast completion: useful to measure latency w/o the sync
int ofi_rmem_post_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_start_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_complete_fast(const int ttl_data, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_wait_fast(const int ncalls, ofi_rmem_t* mem, ofi_comm_t* comm);

// signal
int ofi_rmem_sig_wait(const uint32_t val, ofi_rmem_t* mem);

// operation creation
int ofi_rma_put_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
int ofi_rma_rput_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
int ofi_rma_put_signal_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
// operation management
int ofi_rma_start(ofi_rmem_t* mem, ofi_rma_t* rma);
int ofi_rma_free(ofi_rma_t* rma);

#endif
