/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include <inttypes.h>
#include <stdint.h>
#include <unistd.h>

#include "ofi.h"
#include "ofi_utils.h"
#include "pmi_utils.h"
#include "rdma/fi_atomic.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_rma.h"
#include "rmem_utils.h"

#define m_get_rx(i, mem) (i % mem->ofi.n_rx)

#ifndef NDEBUG
#define m_mem_check_empty_cq(cq)                                                          \
    do {                                                                            \
        ofi_cq_entry event[1];                                                      \
        int ret = fi_cq_read(cq, event, 1);                      \
        uint8_t* op_ctx = (uint8_t*)event[0].op_context;                            \
        uint8_t kind;                                                               \
        if (ret > 0) {                                                              \
            kind = *((uint8_t*)op_ctx +                                             \
                     (offsetof(ofi_cqdata_t, kind) - offsetof(ofi_cqdata_t, ctx))); \
        }                                                                           \
        m_assert(ret <= 0, "ret = %d, cq is NOT empty, kind = %u", ret, kind);      \
    } while (0)
#else
#define m_mem_check_empty_cq(cq) \
    do {                         \
    } while (0)

#endif

int ofi_rmem_init(ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(!(comm->prov->mode & FI_RX_CQ_DATA), "provider needs FI_RX_CQ_DATA");
    //---------------------------------------------------------------------------------------------
    // reset two atomics for the signals with remote write access only
    for (int i = 0; i < m_rma_n_epoch; ++i) {
        m_countr_init(mem->ofi.sync.epch + i);
    }

    // allocate the counters tracking the number of issued calls
    mem->ofi.sync.icntr = calloc(comm->size, sizeof(countr_t));
    for (int i = 0; i < comm->size; ++i) {
        m_countr_init(mem->ofi.sync.icntr + i);
    }

    //---------------------------------------------------------------------------------------------
    // register the memory given by the user
    m_verb("registering user memory");
    m_rmem_call(ofi_util_mr_reg(mem->buf, mem->count, FI_REMOTE_READ | FI_REMOTE_WRITE, comm,
                                &mem->ofi.mr, NULL, &mem->ofi.base_list));
    // register the signal then
#if (!M_WRITE_DATA)
    m_verb("registering the signal memory");
    mem->ofi.signal.val = malloc(sizeof(uint32_t));
    mem->ofi.signal.inc = malloc(sizeof(uint32_t));
    mem->ofi.signal.val[0] = 0;
    mem->ofi.signal.inc[0] = 1;
    m_rmem_call(ofi_util_mr_reg(mem->ofi.signal.val, sizeof(uint32_t),
                                FI_REMOTE_READ | FI_REMOTE_WRITE, comm, &mem->ofi.signal.mr, NULL,
                                &mem->ofi.signal.base_list));
    m_rmem_call(ofi_util_mr_reg(mem->ofi.signal.inc, sizeof(uint32_t), FI_READ | FI_WRITE, comm,
                                &mem->ofi.signal.mr_local, &mem->ofi.signal.desc_local, NULL));
    // create the remote signal
    struct fi_cntr_attr sig_cntr_attr = {
        .events = FI_CNTR_EVENTS_COMP,
        .wait_obj = FI_WAIT_UNSPEC,
    };
    m_ofi_call(fi_cntr_open(comm->domain, &sig_cntr_attr, &mem->ofi.signal.scntr, NULL));
    m_ofi_call(fi_cntr_set(mem->ofi.signal.scntr, 0));
#endif
    //---------------------------------------------------------------------------------------------
    // open shared completion and remote counter
    // remote counter
#if (M_SYNC_RMA_EVENT)
    struct fi_cntr_attr cntr_attr = {
        .events = FI_CNTR_EVENTS_COMP,
        .wait_obj = FI_WAIT_UNSPEC,
    };
    // remote counters - count the number of fi_write/fi_read targeted to me
    m_ofi_call(fi_cntr_open(comm->domain, &cntr_attr, &mem->ofi.rcntr, NULL));
    m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
#endif
    //---------------------------------------------------------------------------------------------
    // allocate one Tx/Rx endpoint per thread context, they all share the transmit queue of the
    // thread the first n_rx endpoints will be Transmit and Receive, the rest is Transmit only
    mem->ofi.n_rx = 1;
    mem->ofi.n_tx = comm->n_ctx;
    const int n_ttl_trx = comm->n_ctx + 1;
    m_assert(mem->ofi.n_rx <= mem->ofi.n_tx, "number of rx must be <= number of tx");
    // allocate n_ctx + 1 structs and get the the right pointer ids
    ofi_rma_trx_t* trx = calloc(n_ttl_trx, sizeof(ofi_rma_trx_t));
    mem->ofi.data_trx = trx + 0;
    mem->ofi.sync_trx = trx + comm->n_ctx;
    for (int i = 0; i < n_ttl_trx; ++i) {
        const bool is_rx = (i < mem->ofi.n_rx);
        const bool is_tx = (i < mem->ofi.n_tx);
        const bool is_sync = (i == comm->n_ctx);

        // ------------------- endpoint
        if (is_rx) {
            // locally copy the srx address, might be overwritten if needed
            trx[i].srx = comm->ctx[i].srx;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &trx[i].ep,
                                        &comm->ctx[i].stx, &trx[i].srx));
        } else if (is_tx) {
            mem->ofi.data_trx[i].srx = NULL;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &trx[i].ep,
                                        &comm->ctx[i].stx, &trx[i].srx));
        } else {
            // thread 0 will do the sync
            trx[i].srx = comm->ctx[0].srx;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &trx[i].ep,
                                        &comm->ctx[0].stx, &trx[i].srx));
        }

        // ------------------- address vector
        if (is_rx || is_sync) {
            m_verb("creating a new AV and binding it");
            // if we create a receive context as well, then get the AV
            struct fi_av_attr av_attr = {
                .type = FI_AV_TABLE,
                .name = NULL,
                .count = comm->size,
            };
            m_ofi_call(fi_av_open(comm->domain, &av_attr, &trx[i].av, NULL));
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].av->fid, 0));
        } else {
            // bind the AV from the corresponding Receive endpoint, otherwise we cannot use it on it
            // because we first build the receive context we are certain that the AV exists
            m_verb("binding EP #%d to AV %d", i, m_get_rx(i, mem));
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[m_get_rx(i, mem)].av->fid, 0));
        }

        // ------------------- completion queue
        struct fi_cq_attr cq_attr = {
            .format = OFI_CQ_FORMAT,
            .wait_obj = FI_WAIT_NONE,
        };
        m_ofi_call(fi_cq_open(comm->domain, &cq_attr, &trx[i].cq, NULL));
        uint64_t tcq_trx_flags = FI_TRANSMIT | FI_RECV;
        m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].cq->fid, tcq_trx_flags));

        // ------------------- bind the counters and the MR
        // if MR_ENDPOINT we have to enable and then bind the MR
        if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            // enable the EP
            m_ofi_call(fi_enable(trx[i].ep));
        }
        if (is_rx) {
#if (M_SYNC_RMA_EVENT)
            m_rmem_call(ofi_util_mr_bind(trx[i].ep, mem->ofi.mr, mem->ofi.rcntr, comm));
#if (!M_WRITE_DATA)
            m_rmem_call(ofi_util_mr_bind(trx[i].ep, mem->ofi.signal.mr_local, NULL, comm));
            m_rmem_call(
                ofi_util_mr_bind(trx[i].ep, mem->ofi.signal.mr, mem->ofi.signal.scntr, comm));
#endif
#else
            m_rmem_call(ofi_util_mr_bind(trx[i].ep, mem->ofi.mr, NULL, comm));
#if (!M_WRITE_DATA)
            m_rmem_call(ofi_util_mr_bind(trx[i].ep, mem->ofi.signal.mr_local, NULL, comm));
            m_rmem_call(
                ofi_util_mr_bind(trx[i].ep, mem->ofi.signal.mr, mem->ofi.signal.scntr, comm));
#endif
#endif
        }
        // is not MR_ENDPOINT, first bind and then enable
        if (!(comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT)) {
            // enable the EP
            m_ofi_call(fi_enable(trx[i].ep));
        }
        if (is_rx || is_sync) {
            // get the addresses from others
            m_rmem_call(ofi_util_av(comm->size, trx[i].ep, trx[i].av, &trx[i].addr));
        }
        m_verb("done with EP # %d", i);
    }

    //---------------------------------------------------------------------------------------------
    // if needed, enable the MR and then get the corresponding key and share it
    // first the user region's key
    m_rmem_call(ofi_util_mr_enable(mem->ofi.mr, comm, &mem->ofi.key_list));
#if (!M_WRITE_DATA)
    m_rmem_call(ofi_util_mr_enable(mem->ofi.signal.mr, comm, &mem->ofi.signal.key_list));
    m_rmem_call(ofi_util_mr_enable(mem->ofi.signal.mr_local, comm, NULL));
#endif

    //---------------------------------------------------------------------------------------------
    // allocate the data user for sync
    // we need one ctx entry per rank
    // all of them refer to the same sync epoch
    mem->ofi.sync.cqdata = calloc(comm->size, sizeof(ofi_cqdata_t));
    for (int i = 0; i < comm->size; ++i) {
        ofi_cqdata_t* ccq = mem->ofi.sync.cqdata + i;
        ccq->kind = m_ofi_cq_kind_sync;
        ccq->sync.cntr = mem->ofi.sync.epch;
        m_rmem_call(ofi_util_mr_reg(&ccq->sync.buf, sizeof(uint64_t), FI_SEND | FI_RECV, comm,
                                    &ccq->sync.buf_mr, &ccq->sync.buf_desc, NULL));
        m_rmem_call(ofi_util_mr_bind(mem->ofi.sync_trx->ep, ccq->sync.buf_mr, NULL, comm));
        m_rmem_call(ofi_util_mr_enable(ccq->sync.buf_mr, comm, NULL));
        // if(comm->prov->domain_attr->mr_mode & FI_MR_LOCAL){
        //     m_assert(comm->prov->domain_attr->mr_mode & FI_MR_PROV_KEY,"we assume prov key here");
        //     uint64_t access = FI_SEND | FI_RECV;
        //     uint64_t sync_flags = 0;
        //     m_ofi_call(fi_mr_reg(comm->domain, &ccq->sync.buf, sizeof(uint64_t), access, 0,
        //                          comm->unique_mr_key++, sync_flags, &ccq->sync.buf_mr, NULL));
        //     ccq->sync.buf_desc = fi_mr_desc(ccq->sync.buf_mr);
        // } else {
        //     ccq->sync.buf_mr = NULL;
        //     ccq->sync.buf_desc = NULL;
        // }
    }
    return m_success;
}
int ofi_rmem_free(ofi_rmem_t* mem, ofi_comm_t* comm) {
    struct fid_ep* nullsrx = NULL;
    struct fid_stx* nullstx = NULL;
    // free the sync
    for (int i = 0; i < comm->size; ++i) {
        m_rmem_call(ofi_util_mr_close(mem->ofi.sync.cqdata[i].sync.buf_mr));
    }
    free(mem->ofi.sync.cqdata);
    // free the MRs
    m_rmem_call(ofi_util_mr_close(mem->ofi.mr));
    free(mem->ofi.key_list);
    free(mem->ofi.base_list);
#if (!M_WRITE_DATA)
    m_rmem_call(ofi_util_mr_close(mem->ofi.signal.mr));
    m_rmem_call(ofi_util_mr_close(mem->ofi.signal.mr_local));
    m_ofi_call(fi_close(&mem->ofi.signal.scntr->fid));
    free(mem->ofi.signal.key_list);
    free(mem->ofi.signal.base_list);
    free(mem->ofi.signal.val);
    free(mem->ofi.signal.inc);
#endif

    // free the Tx first, need to close them before closing the AV in the Rx
    const int n_trx = comm->n_ctx + 1;
    for (int i = 0; i < n_trx; ++i) {
        const bool is_rx = (i < mem->ofi.n_rx);
        const bool is_tx = (i < mem->ofi.n_tx);
        const bool is_sync = (i == comm->n_ctx);

        ofi_rma_trx_t* trx = (is_sync) ? (mem->ofi.sync_trx) : (mem->ofi.data_trx + i);
        m_rmem_call(ofi_util_free_ep(comm->prov, &trx->ep, &nullstx, &nullsrx));
        m_ofi_call(fi_close(&trx->cq->fid));
        if (is_rx || is_sync) {
            m_ofi_call(fi_close(&trx->av->fid));
            free(trx->addr);
        }
    }
#if (M_SYNC_RMA_EVENT)
    m_ofi_call(fi_close(&mem->ofi.rcntr->fid));
#endif
    free(mem->ofi.data_trx);
    // free the sync
    free(mem->ofi.sync.icntr);
    return m_success;
}

typedef enum {
    RMA_OPT_PUT,
    RMA_OPT_RPUT,
    RMA_OPT_PUT_SIG,
} rma_opt_t;

static int ofi_rma_init(ofi_rma_t* rma, ofi_rmem_t* mem, const int ctx_id, ofi_comm_t* comm,
                        rma_opt_t op) {
    m_assert(ctx_id < comm->n_ctx, "ctx id = %d < the number of ctx = %d", ctx_id, comm->n_ctx);
    //----------------------------------------------------------------------------------------------
    // endpoint and address
    const int rx_id = m_get_rx(ctx_id, mem);
    rma->ofi.ep = mem->ofi.data_trx[ctx_id].ep;
    rma->ofi.addr = mem->ofi.data_trx[rx_id].addr[rma->peer];
    //----------------------------------------------------------------------------------------------
    // if needed, register the memory
    m_rmem_call(ofi_util_mr_reg(rma->buf, rma->count, FI_WRITE | FI_READ, comm,
                                &rma->ofi.msg.mr_local, &rma->ofi.msg.desc_local, NULL));
    m_rmem_call(ofi_util_mr_bind(rma->ofi.ep, rma->ofi.msg.mr_local, NULL, comm));
    m_rmem_call(ofi_util_mr_enable(rma->ofi.msg.mr_local, comm, NULL));
    // if (comm->prov->domain_attr->mr_mode & FI_MR_LOCAL) {
    //     m_assert(comm->prov->domain_attr->mr_mode & FI_MR_PROV_KEY,"prov key is needed here");
    //     uint64_t access = FI_WRITE | FI_READ;
    //     uint64_t flags = 0;
    //     m_ofi_call(fi_mr_reg(comm->domain, rma->buf, rma->count, access, 0, comm->unique_mr_key++,
    //                          flags, &rma->ofi.msg.mr_local, NULL));
    //     rma->ofi.msg.desc_local = fi_mr_desc(rma->ofi.msg.mr_local);
    // } else {
    //     rma->ofi.msg.mr_local = NULL;
    //     rma->ofi.msg.desc_local = NULL;
    // }

    //----------------------------------------------------------------------------------------------
    // IOVs
    rma->ofi.msg.iov = (struct iovec){
        .iov_base = rma->buf,
        .iov_len = rma->count,
    };
    m_verb("rma-init: base = %llu + disp = %lu", mem->ofi.base_list[rma->peer], rma->disp);
    rma->ofi.msg.riov = (struct fi_rma_iov){
        .addr = mem->ofi.base_list[rma->peer] + rma->disp,  // offset from key
        .len = rma->count,
        .key = mem->ofi.key_list[rma->peer],
    };
    // cq and progress
    // any of the cqdata entry can be used to fallback, the first one always exists
    rma->ofi.progress.cq = mem->ofi.data_trx[ctx_id].cq;
    rma->ofi.progress.fallback_ctx = &mem->ofi.sync.cqdata[0].ctx;
    switch (op) {
        case (RMA_OPT_PUT): {
            rma->ofi.msg.cq.kind = m_ofi_cq_inc_local | m_ofi_cq_kind_null;
        } break;
        case (RMA_OPT_RPUT): {
            rma->ofi.msg.cq.kind = m_ofi_cq_inc_local | m_ofi_cq_kind_rqst;
            m_countr_store(&rma->ofi.msg.cq.rqst.busy, 1);
        } break;
        case (RMA_OPT_PUT_SIG): {
            rma->ofi.msg.cq.kind = m_ofi_cq_inc_local | m_ofi_cq_kind_null;
        } break;
    }
    // flag
    const bool auto_progress = (comm->prov->domain_attr->data_progress & FI_PROGRESS_AUTO);
    const bool do_inject = (rma->count < comm->prov->tx_attr->inject_size) && auto_progress;
    rma->ofi.msg.flags = (do_inject ? FI_INJECT : 0x0);
    switch (op) {
        case (RMA_OPT_PUT): {
            rma->ofi.msg.flags |= (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE);
        } break;
        case (RMA_OPT_RPUT): {
            rma->ofi.msg.flags |= (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE);
            rma->ofi.msg.flags |= FI_COMPLETION;
        } break;
        case (RMA_OPT_PUT_SIG): {
#if (M_WRITE_DATA)
            rma->ofi.msg.flags |= (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE);
#else
            rma->ofi.msg.flags |= FI_DELIVERY_COMPLETE;
#endif
        } break;
    }
        // if we don't use the remote event, need to use FI_REMOTE_CQ_DATA
#if (!M_SYNC_RMA_EVENT)
    rma->ofi.msg.flags |= FI_REMOTE_CQ_DATA;
#endif

    //----------------------------------------------------------------------------------------------
#if (M_WRITE_DATA)
    if (op == RMA_OPT_PUT_SIG) {
        rma->ofi.msg.flags |= FI_REMOTE_CQ_DATA;
        rma->ofi.sig.data = m_ofi_data_set_sig;
    } else {
        rma->ofi.sig.data = 0x0;
    }
#else
    if (op == RMA_OPT_PUT_SIG) {
        // iovs
        rma->ofi.sig.iov = (struct fi_ioc){
            .addr = mem->ofi.signal.inc,
            .count = 1,
        };
        rma->ofi.sig.riov = (struct fi_rma_ioc){
            .addr = mem->ofi.signal.base_list[rma->peer] + 0,  // offset from key
            .count = 1,
            .key = mem->ofi.signal.key_list[rma->peer],
        };
        // setup cq data
        rma->ofi.sig.cq.kind = m_ofi_cq_inc_local | m_ofi_cq_kind_null;
        // flag
        rma->ofi.sig.flags = FI_FENCE | (do_inject ? FI_INJECT : 0x0) |
                             (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE);
    } else {
        rma->ofi.sig.flags = 0x0;
        rma->ofi.sig.iov = (struct fi_ioc){0};
        rma->ofi.sig.riov = (struct fi_rma_ioc){0};
    }
#endif
    //----------------------------------------------------------------------------------------------
    m_assert(rma->ofi.msg.riov.key != FI_KEY_NOTAVAIL, "key must be >0");
    return m_success;
}
int ofi_put_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_PUT);
}
int ofi_rput_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_RPUT);
}
int ofi_put_signal_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_PUT_SIG);
}

int ofi_rma_start(ofi_rmem_t* mem, ofi_rma_t* rma) {
    //----------------------------------------------------------------------------------------------
    uint64_t flags = rma->ofi.msg.flags;
    struct fi_msg_rma msg = {
        .msg_iov = &rma->ofi.msg.iov,
        .desc = &rma->ofi.msg.desc_local,
        .iov_count = 1,
        .addr = rma->ofi.addr,
        .rma_iov = &rma->ofi.msg.riov,
        .rma_iov_count = 1,
        .data = 0x0,
        .context = &rma->ofi.msg.cq.ctx,
    };
#if (!M_SYNC_RMA_EVENT)
    msg.data |= m_ofi_data_set_rcq;
#endif
#if (M_WRITE_DATA)
    msg.data |= rma->ofi.sig.data;
#endif
    m_verb("write msg with kind =%d (inc local? %d) to ep %p", rma->ofi.msg.cq.kind & 0x0f,
           rma->ofi.msg.cq.kind & m_ofi_cq_inc_local, rma->ofi.ep);
    m_verb("doing it");
    m_ofi_call_again(fi_writemsg(rma->ofi.ep, &msg, flags),&rma->ofi.progress);
    //----------------------------------------------------------------------------------------------
#if (!M_WRITE_DATA)
    if (rma->ofi.sig.flags) {
        struct fi_msg_atomic sigmsg = {
            .msg_iov = &rma->ofi.sig.iov,
            .desc = &mem->ofi.signal.desc_local,
            .iov_count = 1,
            .addr = rma->ofi.addr,
            .rma_iov = &rma->ofi.sig.riov,
            .rma_iov_count = 1,
            .datatype = FI_INT32,
            .op = FI_SUM,
            .data = 0x0,  // atomics does NOT support FI_REMOTE_CQ_DATA
            .context = &rma->ofi.sig.cq.ctx,
        };
        m_ofi_call_again(fi_atomicmsg(rma->ofi.ep, &sigmsg, rma->ofi.sig.flags),
                         &rma->ofi.progress);
    }
#endif
    //----------------------------------------------------------------------------------------------
    // increment the counter
    // always update by one as the target will NOT wait for the atomic to complete, it's a separate
    // completion process
    m_countr_fetch_add(&mem->ofi.sync.icntr[rma->peer], 1);
#if (!M_WRITE_DATA)
    if (rma->ofi.sig.flags) {
        // if we do a signal call we need to wait for its completion BUT we don't send that information to the target
        m_countr_fetch_add(&mem->ofi.sync.isig, 1);
    }
#endif
    // #if (M_WRITE_DATA)
    //     m_countr_fetch_add(&mem->ofi.sync.icntr[rma->peer], 1);
    // #else
    //     m_countr_fetch_add(&mem->ofi.sync.icntr[rma->peer], (rma->ofi.sig.flags) ? 2 : 1);
    // #endif
    // if we had to get a cq entry and the inject, mark is as done
    if (flags & FI_INJECT && rma->ofi.msg.cq.kind == m_ofi_cq_kind_rqst) {
        m_countr_fetch_add(&rma->ofi.msg.cq.rqst.busy, -1);
    }
    m_assert(rma->ofi.msg.riov.key != FI_KEY_NOTAVAIL, "key must be >0");

    return m_success;
}
int ofi_rma_put_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_PUT);
}
int ofi_rma_rput_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_RPUT);
}
int ofi_rma_put_signal_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_PUT_SIG);
}

int ofi_rma_free(ofi_rma_t* rma) {
    m_rmem_call(ofi_util_mr_close(rma->ofi.msg.mr_local));
    return m_success;
}

// notify the processes in comm of memory exposure epoch
int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    m_verb("posting");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    // no call access in my memory can be done before the notification, it's safe to reset the
    // counters involved in the memory exposure: epoch[1:2]
    // do NOT reset the epoch[0], it's already exposed to the world!
    m_countr_store(m_rma_mepoch_cmpl(mem), 0);
    m_countr_store(m_rma_mepoch_remote(mem), 0);

    // notify readiness to the rank list
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata + i;
        cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqdata->sync.buf = m_ofi_data_set_post;
        uint64_t tag = m_ofi_tag_set_ps;
        m_ofi_call_again(
            fi_tsend(mem->ofi.sync_trx->ep, &cqdata->sync.buf, sizeof(uint64_t),
                     cqdata->sync.buf_desc, mem->ofi.sync_trx->addr[rank[i]], tag, &cqdata->ctx),
            &progress);
    }
    //----------------------------------------------------------------------------------------------
    // wait for completion of the send calls
    while (m_countr_load(m_rma_mepoch_local(mem)) < nrank) {
        m_rmem_call(ofi_progress(&progress));
    }
    m_countr_fetch_add(m_rma_mepoch_local(mem), -nrank);
    // cq must be empty now
    m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    m_verb("posted");
    return m_success;
}

// wait for the processes in comm to notify their exposure
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    m_verb("starting");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    // open the handshake requests
    struct iovec iov = {
        .iov_len = sizeof(uint64_t),
    };
    struct fi_msg_tagged msg = {
        .msg_iov = &iov,
        .iov_count = 1,
        .tag = m_ofi_tag_set_ps,
        .ignore = 0x0,
        .data = 0,
    };
    // all the ctx of sync.cqdata will lead to the same epoch array
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata+i;
        cqdata->kind = m_ofi_cq_kind_sync;
        iov.iov_base = &cqdata->sync.buf;
        msg.desc = &cqdata->sync.buf_desc;
        msg.context = &cqdata->ctx;
        msg.addr = mem->ofi.sync_trx->addr[rank[i]];
        uint64_t flags = FI_COMPLETION;
        m_ofi_call_again(fi_trecvmsg(mem->ofi.sync_trx->srx, &msg, flags),&progress);
    }
    // while waiting for the receive, reset the value of icntr. 
    // It's needed if we use the fast completion mechanism
    for (int i = 0; i < nrank; ++i) {
        m_countr_exchange(&mem->ofi.sync.icntr[rank[i]], 0);
    }
    // wait for completion, recv are NOT tracked by ccntr
    do {
        ofi_progress(&progress);
    } while (m_countr_load(m_rma_mepoch_post(mem)) < nrank);
    m_countr_fetch_add(m_rma_mepoch_post(mem), -nrank);
    // the sync cq MUST be empty now
    m_verb("started");
    m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    return m_success;
}

int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("completing");
    // count the number of calls issued for each of the ranks and notify them
    int ttl_data = 0;
    uint64_t tag = m_ofi_tag_set_cw;
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
    for (int i = 0; i < nrank; ++i) {
        int issued_rank = m_countr_exchange(&mem->ofi.sync.icntr[rank[i]], 0);
        ttl_data += issued_rank;
        // notify
        ofi_rma_trx_t* trx = mem->ofi.sync_trx;
        ofi_cqdata_t* cqd = mem->ofi.sync.cqdata + i;
        cqd->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqd->sync.buf = m_ofi_data_set_cmpl | m_ofi_data_set_nops(issued_rank);
        m_ofi_call_again(fi_tsend(trx->ep, &cqd->sync.buf, sizeof(uint64_t), cqd->sync.buf_desc,
                                  trx->addr[rank[i]], tag, &cqd->ctx),
                         &progress);
    }
    //----------------------------------------------------------------------------------------------
    // count the number of completed calls and wait till they are all done
    // must complete all the sync call done in rmem_post (if any) + the signal calls
#if (!M_WRITE_DATA)
    m_verb("complete: waiting for %d syncs, %d calls and %d signals to complete", nrank, ttl_data,
           m_countr_load(&mem->ofi.sync.isig));
    uint64_t threshold = nrank + ttl_data + m_countr_exchange(&mem->ofi.sync.isig, 0);
#else
    uint64_t threshold = nrank + ttl_data;
    m_verb("complete: waiting for %d syncs, %d calls (total: %" PRIu64 ")", nrank, ttl_data,
           threshold);
#endif
    // we loop on data_trx array using the global length (n_tx+1) to progress everybody equally,
    // starting with the sync (index n_tx)
    int i = mem->ofi.n_tx;
    while (m_countr_load(m_rma_mepoch_local(mem)) < threshold) {
        progress.cq = mem->ofi.data_trx[i].cq;
        m_ofi_call(ofi_progress(&progress));
        i = (i + 1) % (mem->ofi.n_tx + 1);
    }
    m_countr_fetch_add(m_rma_mepoch_local(mem), -threshold);

#ifndef NDEBUG
    m_verb("completed");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("waitting");
    //----------------------------------------------------------------------------------------------
    // ideally we can pre-post them, but then the fast completion would have to cancel them
    struct iovec iov = {
        .iov_len = sizeof(uint64_t),
    };
    struct fi_msg_tagged msg = {
        .msg_iov = &iov,
        .iov_count = 1,
        .tag = m_ofi_tag_set_cw,
        .ignore = 0x0,
        .data = 0,
    };
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
    uint64_t flags = FI_COMPLETION;
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata + i;
        cqdata->kind = m_ofi_cq_kind_sync;
        iov.iov_base = &cqdata->sync.buf;
        msg.desc = &cqdata->sync.buf_desc;
        msg.context = &cqdata->ctx;
        msg.addr = mem->ofi.sync_trx->addr[rank[i]];
        m_ofi_call_again(fi_trecvmsg(mem->ofi.sync_trx->srx, &msg, flags),&progress);
    }
    //----------------------------------------------------------------------------------------------
    // get the number of calls done by the origins
    int i = 0;
    while (m_countr_load(m_rma_mepoch_cmpl(mem)) < nrank) {
        // every try progress the sync, we really need it!
        progress.cq = mem->ofi.sync_trx->cq;
        ofi_progress(&progress);

        // progress the data as well while we are at it
        progress.cq = mem->ofi.data_trx[i].cq;
        ofi_progress(&progress);

        // update the counter to loop on the data receive trx
        i = (i + 1) % mem->ofi.n_rx;
    }
    m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    // now we know for how many calls we have to wait, all the RECV have completed at this point
    m_verb("waitall: waiting for %d calls to complete", m_countr_load(m_rma_mepoch_remote(mem)));
#if (M_SYNC_RMA_EVENT)
    uint64_t threshold = m_countr_load(m_rma_mepoch_remote(mem));
    // the counter is linked to the MR so waiting on it will trigger progress
    m_verb("current value of rcntr: %llu",fi_cntr_read(mem->ofi.rcntr));
    m_ofi_call(fi_cntr_wait(mem->ofi.rcntr, threshold, -1));
    m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
#else
    // every put comes with data that will substract 1 to the epoch[2] value
    // no need to progress the sync_trx here!
    i = 0;
    while (m_countr_load(m_rma_mepoch_remote(mem)) > 0) {
        progress.cq = mem->ofi.data_trx[i].cq;
        ofi_progress(&progress);
        // update the counter, loop on the reception trx only
        i = (i + 1) % mem->ofi.n_rx;
    }
    // counter cannot be <0, that would indicate that too many cq entries have been processed
    m_assert(m_countr_load(m_rma_mepoch_remote(mem)) == 0, "counter cannot be %d",m_countr_load(m_rma_mepoch_remote(mem)));
#endif
#ifndef NDEBUG
    m_verb("completed");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

int ofi_rmem_complete_fast(const int ttl_data, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("completing -fast");
    //----------------------------------------------------------------------------------------------
#if (!M_WRITE_DATA)
    m_assert(m_countr_load(&mem->ofi.sync.isig) == 0,
             "cannot use the complete fast when signal have been issued with fi_atomic");
#endif
    uint64_t threshold = ttl_data;
    // rma calls generate cq entries so they need to be processed, we loop on the data_trx only
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
    int i = 0;
    while (m_countr_load(m_rma_mepoch_local(mem)) < threshold) {
        progress.cq = mem->ofi.data_trx[i].cq;
        m_ofi_call(ofi_progress(&progress));
        i = (i + 1) % (mem->ofi.n_tx);
    }
    m_countr_fetch_add(m_rma_mepoch_local(mem), -threshold);

#ifndef NDEBUG
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

int ofi_rmem_wait_fast(const int ncalls, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("waiting fast");
    // compare the number of calls done to the value in the epoch if everybody has finished
    m_verb("wait untill: waiting for %d calls to complete", ncalls);
#if (M_SYNC_RMA_EVENT)
    // the counter is linked to the MR so waiting on it will trigger progress
    m_ofi_call(fi_cntr_wait(mem->ofi.rcntr, ncalls, -1));
    m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
#else
    // every put comes with data that will substract 1 to the epoch[2] value
    // first bump the value of epoch[2]
    m_countr_fetch_add(m_rma_mepoch_remote(mem), ncalls);
    // wait for it to come down
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
    int i = 0;
    while (m_countr_load(m_rma_mepoch_remote(mem)) > 0) {
        progress.cq = mem->ofi.data_trx[i].cq;
        ofi_progress(&progress);
        // update the counter
        i = (i + 1) % mem->ofi.n_tx;
    }
    // a negative counter is valid as it is possible that many cq entries are processed but the user
    // waits only for a few of them
#endif
#ifndef NDEBUG
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

// end of file
