/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef OFI_H_
#define OFI_H_
#include <assert.h>
#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "rmem.h"

// gpu specific
#include "gpu.h"


#define m_ofi_cq_entries     16
#define m_ofi_am_max_size    8  // uint64_t
#define m_ofi_am_buf_num     2
#define m_ofi_am_buf_size    (m_ofi_cq_entries * m_ofi_am_max_size)
#define m_ofi_am_cq_min_size m_ofi_am_max_size

#define OFI_CQ_FORMAT FI_CQ_FORMAT_DATA
typedef struct fi_cq_data_entry ofi_cq_entry;

//--------------------------------------------------------------------------------------------------
#ifndef USE_MR_LOCAL
#define M_FORCE_MR_LOCAL 0
#else
#define M_FORCE_MR_LOCAL 1
#endif

#ifndef USE_ASYNC_PROGRESS
#define M_FORCE_ASYNC_PROGRESS 0
#else
#define M_FORCE_ASYNC_PROGRESS 1
#endif

//--------------------------------------------------------------------------------------------------
#define likely(x_)   __builtin_expect(!!(x_),1)
#define unlikely(x_) __builtin_expect(!!(x_),0)

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

#define m_ofi_data_set_rcq     ((uint64_t)0x1 << m_ofi_data_bit_rcq)
#define m_ofi_data_set_sig     ((uint64_t)0x1 << m_ofi_data_bit_sig)
#define m_ofi_data_set_post    ((uint64_t)0x1 << m_ofi_data_bit_post)
#define m_ofi_data_set_cmpl    ((uint64_t)0x1 << m_ofi_data_bit_cmpl)
#define m_ofi_data_get_rcq(a)  ((a >> m_ofi_data_bit_rcq) & 0x1)
#define m_ofi_data_get_post(a) ((a >> m_ofi_data_bit_post) & 0x1)
#define m_ofi_data_get_cmpl(a) ((a >> m_ofi_data_bit_cmpl) & 0x1)
#define m_ofi_data_mask_nops   (((uint64_t)0x1 << m_ofi_data_avail) - 1)
#define m_ofi_data_set_nops(a) (((uint64_t)a) & m_ofi_data_mask_nops)
#define m_ofi_data_get_nops(a) ((uint32_t)(a & m_ofi_data_mask_nops))

//--------------------------------------------------------------------------------------------------
// if true need to bump the local counter
#define m_ofi_cq_inc_local (0x80)  // 1000 0000
// define the opeation for after
#define m_ofi_cq_kind_null (0x01)  // 0000 0001
#define m_ofi_cq_kind_sync (0x02)  // 0000 0010
#define m_ofi_cq_kind_rqst (0x04)  // 0000 0100
#define m_ofi_cq_kind_am   (0x08)  // 0000 1000

//--------------------------------------------------------------------------------------------------
/**
 * @brief define remote completion mode for RMA
 */
typedef enum {
    M_OFI_RCMPL_NULL,
    M_OFI_RCMPL_FENCE,
    M_OFI_RCMPL_CQ_DATA,
    M_OFI_RCMPL_DELIV_COMPL,
    M_OFI_RCMPL_REMOTE_CNTR,
} ofi_rcmpl_mode_t;
/**
 * @brief define ready-to-receive mode for RMA (post-start sync)
 */
typedef enum {
    M_OFI_RTR_NULL,
    M_OFI_RTR_MSG,
    M_OFI_RTR_TAGGED,
    M_OFI_RTR_ATOMIC,
} ofi_rtr_mode_t;
/**
 * @brief define down-to-close mode for RMA (complete-wait sync)
 */
typedef enum {
    M_OFI_DTC_NULL,
    M_OFI_DTC_MSG,
    M_OFI_DTC_TAGGED,
} ofi_dtc_mode_t;
/**
 * @brief operational mode: define how the provider is going to operate
 */
typedef struct {
    ofi_rtr_mode_t rtr_mode;
    ofi_dtc_mode_t dtc_mode;
    ofi_rcmpl_mode_t rcmpl_mode;
} ofi_mode_t;

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

    uint64_t unique_mr_key;  // used for MR
    struct fi_info* prov;
    struct fid_fabric* fabric;
    struct fid_domain* domain;

    ofi_mode_t prov_mode;  //!< supported provider's operational modes
    //
    int n_ctx;
    ofi_ctx_t* ctx;
} ofi_comm_t;

/**
 * @brief ofi memory buffer for local memory
 */
typedef struct {
    void* desc;        //!< memory descriptor
    struct fid_mr* mr;  //!< memory registration handle
} ofi_local_mr_t;

/**
 * @brief ofi memory buffer for remote memory
 */
typedef struct {
    uint64_t* base_list;  //!< list of base offsets (typically 0, unless FI_MR_VIRT_ADDR)
    uint64_t* key_list;
    struct fid_mr* mr;  //!< memory registration handle
} ofi_remote_mr_t;

/**
 * @brief ofi active message buffer
 */
typedef struct {
    void* buf;
    ofi_local_mr_t mr;
} ofi_am_buf_t;

/**
 * @brief data-structure used in the cq when an operation has completed
 *
 * The cq entry will return the address to the context.
 * From there we can offset memory location to access the kind argument and perform different
 * operations
 */
typedef struct {
    // fi_context mandatory for each CQ entry
    struct fi_context ctx;
    // kind parameter
    uint8_t kind;
    union {
        struct {
            countr_t busy;  //!< completed if 0
        } rqst;             //!< kind == m_ofi_cq_kind_rqst
        struct {
            uint64_t data;        //!< actual buffer memory
            ofi_local_mr_t mr;    //!< local memory registration
        } sync;                   // kind == m_ofi_cq_kind_sync
        struct {
            ofi_am_buf_t* buf;
            struct fid_ep* srx;
        } am;
    };
    // array of epochs
    countr_t* epoch_ptr;  //!< array of epochs needed when m_ofi_cq_kind_local is added to the kind
                          //!< argument (compatible with every other kind of request)
} ofi_cqdata_t;

/**
 * @brief special context, used in the cq when we have to handle special operations:
 * - cq data is received: no ctx is available
 */
typedef struct {
    countr_t* epoch_ptr;
} ofi_special_cq_t;

/**
 * @brief used to read the ofi_cq entry
 *
 */
typedef struct {
    struct fid_cq* cq;  //!< cq to progress
    // /**
    //  * @brief fallback context pointer, used if the ctx received is NULL, aka the entry is a
    //  * REMOTE_CQ_DATA. Value cannot be null if used explicitely to progress
    //  */
    // // void* fallback_ctx;
    ofi_special_cq_t xctx; //!< special context pointer, used for special treatment
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
        ofi_local_mr_t mr;
        // void* desc_local;
        // struct fid_mr* mr_local;
        // iovs
        struct iovec iov;
        struct fi_msg_tagged msg;
        // endpoint
        p2p_opt_t kind;
        uint64_t flags;
        struct fid_ep* ep;
    } ofi;
} ofi_p2p_t;

/**
 * @brief transmit and receive context. meant to be per-thread
 */
typedef struct {
    fi_addr_t* addr;     //!< address list (comm world rank)
    struct fid_ep* ep;   //!< endpoint
    struct fid_ep* srx;  //!< receive endpoint
    struct fid_cq* cq;   //!< completion queue for RECEIVE and REMOTE_DATA
    struct fid_av* av;   //!< address vector
    gpuStream_t stream;
} ofi_rma_trx_t;

typedef struct {
    uint32_t inc;  //!< increment, always 1
    uint32_t val;  //!< exposed value
    uint32_t res;  //!< local value to red the exposed one
    ofi_local_mr_t inc_mr;
    ofi_local_mr_t res_mr;
    ofi_remote_mr_t val_mr;
    ofi_cqdata_t read_cqdata; //!< cqdata used to read the value
} ofi_mem_sig_t;


#define m_rma_epoch_post(e)    (e + 0)                 // posted
#define m_rma_epoch_cmpl(e)    (e + 1)                 // completed
#define m_rma_epoch_remote(e)  (e + 2)                 // remote fi_write
#define m_rma_epoch_local(e)   (e + 3)                 // local (tsend, fi_write/read)
#define m_rma_mepoch_post(m)   (m->ofi.sync.epch + 0)  // posted
#define m_rma_mepoch_cmpl(m)   (m->ofi.sync.epch + 1)  // completed
#define m_rma_mepoch_remote(m) (m->ofi.sync.epch + 2)  // remote fi_write
#define m_rma_mepoch_local(m)  (m->ofi.sync.epch + 3)  // local (tsend, fi_write/read)
#define m_rma_n_epoch          4

/**
 * @brief sync data structure
 *
 * Note:
 * - we need different cqdata_t for PS and CW to be able to prepost the CW handshake
 * - the cqdata_ps and cqdata_cw are used mostly to issue messages. only when using tagged messages,
 * they are used to receive them as well. This can be an issue if the "post" step has not completed
 * before the "start" step if a process has to do them both
 */
typedef struct {
    countr_t epch[m_rma_n_epoch];
    countr_t* icntr;       //!< array of fi_write counter (for each rank)
    ofi_mem_sig_t ps_sig;  //!< atomic sync for RTR
    struct {
        ofi_am_buf_t* buf;     //!< active messaging buffer
        ofi_cqdata_t* cqdata;  //!< completion data for the FI_MULTI_RECV
    } am;                      // both for RTR and DTC
    ofi_cqdata_t* cqdata_ps;  //!< completion data for each rank Post-Start
    ofi_cqdata_t* cqdata_cw;  //!< completion data for each rank Complete-Wait
} ofi_rma_sync_t;

typedef struct {
    // user provided information
    void* buf;     // address of the buffer
    size_t count;  // count in bytes
    int peer;      // destination/origin rank
    ssize_t disp;  // displacement compared to the rmem base ptr (in bytes)

    // implementation specifics
    struct {
        rmem_qnode_t qnode;
        ofi_progress_t progress;  // used to trigger progress
        // data description and ofi msg
        struct {
            uint64_t flags;
            uint64_t data;
            ofi_local_mr_t mr;
            // iovs
            struct iovec iov;
            struct fi_rma_iov riov;
            ofi_cqdata_t cq;
        } msg;
        fi_addr_t addr;
        struct fid_ep* ep;
        gpuStream_t* stream;
    } ofi;
} ofi_rma_t;

// thread-based data structure, must stay valid accross the thread execution
typedef struct {
    countr_t* do_progress;  //<! acts like a switch to activate or not the  progress in the list
    rmem_qmpsc_t* workq;
    ofi_rma_trx_t* data_trx;
    ofi_special_cq_t xctx;
    int n_tx;
} rmem_thread_arg_t;

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
        ofi_remote_mr_t mr;
        // transmit and receive contexts
        ofi_rma_trx_t* sync_trx;
        ofi_rma_trx_t* data_trx;

        // completion and remote counter global for all trx
        struct fid_cntr* rcntr;  // Completed CouNTeR put and get
        // synchronization (PSCW)
        ofi_rma_sync_t sync;
        // work queue
        pthread_t progress;
        rmem_qmpsc_t qtrigr;
        rmem_thread_arg_t thread_arg;
        // gpu triggered resources
        countr_t trigr_count;
        rmem_trigr_ptr h_trigr_pool;
        rmem_trigr_ptr d_trigr_pool;
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
// trig
void* ofi_tthread_main(void* arg);

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

// operation creation
int ofi_rma_put_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
int ofi_rma_rput_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm);
// operation management
int ofi_rma_enqueue(ofi_rmem_t* mem, ofi_rma_t* rma, rmem_trigr_ptr* trigr, rmem_device_t dev);
int ofi_rma_start(ofi_rmem_t* mem, ofi_rma_t* rma, rmem_device_t dev);
int ofi_rma_wait(ofi_rma_t* p2p);
int ofi_rma_free(ofi_rma_t* rma);


int ofi_rma_start_from_task(rmem_qnode_t* task);

#endif
