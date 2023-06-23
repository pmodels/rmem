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

int ofi_rmem_init(ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(!(comm->prov->mode & FI_RX_CQ_DATA), "provider needs FI_RX_CQ_DATA");

    //---------------------------------------------------------------------------------------------
    // reset two atomics for the signals with remote write access only
    m_countr_init(mem->ofi.sync.epoch + 0);
    m_countr_init(mem->ofi.sync.epoch + 1);
    m_countr_init(mem->ofi.sync.epoch + 2);

    // allocate the counters tracking the number of issued calls
    mem->ofi.sync.icntr = calloc(comm->size, sizeof(atomic_int));
    for (int i = 0; i < comm->size; ++i) {
        m_countr_init(mem->ofi.sync.icntr + i);
    }

    //---------------------------------------------------------------------------------------------
    // register the memory given by the user together with the memory used for signaling
    uint64_t flags = 0;
    struct iovec iov;
    struct fi_mr_attr mr_attr = {
        .mr_iov = &iov,
        .iov_count = 1,
        .access = FI_REMOTE_READ | FI_REMOTE_WRITE,
        .offset = 0,
        .requested_key = 0,
        .context = NULL,
    };
    // register the user memory if not NULL
    if (mem->buf && mem->count > 0) {
        iov = (struct iovec){
            .iov_base = mem->buf,
            .iov_len = mem->count,
        };
        m_ofi_call(fi_mr_regattr(comm->domain, &mr_attr, flags, &mem->ofi.mr));
    } else {
        mem->ofi.mr = NULL;
    }
    // register the signal then
    iov = (struct iovec){
        .iov_base = &mem->ofi.signal.val,
        .iov_len = sizeof(mem->ofi.signal.val),
    };
    // we need to request another key if the prov doesn't handle them
    if (!(comm->prov->domain_attr->mr_mode & FI_MR_PROV_KEY)) {
        mr_attr.requested_key++;
    }
    m_ofi_call(fi_mr_regattr(comm->domain, &mr_attr, flags, &mem->ofi.signal.mr));
    // set the signal inc to 1
    mem->ofi.signal.inc = 1;
    //---------------------------------------------------------------------------------------------
    // allocate one Tx/Rx endpoint per thread context, they all share the transmit queue of the
    // thread the first n_rx endpoints will be Transmit and Receive, the rest is Transmit only
    mem->ofi.n_rx = 1;
    mem->ofi.n_tx = comm->n_ctx;
    const int n_ttl_trx = comm->n_ctx + 1;
    m_assert(mem->ofi.n_rx <= mem->ofi.n_tx,"number of rx must be <= number of tx");
    // allocate n_ctx + 1 structs
    ofi_rma_trx_t* trx = calloc(n_ttl_trx, sizeof(ofi_rma_trx_t));
    for (int i = 0; i < n_ttl_trx; ++i) {
        const bool is_rx = (i < mem->ofi.n_rx);
        const bool is_tx = (i < mem->ofi.n_tx);
        const bool is_sync = (i == comm->n_ctx);
        // associate the right trx struct
        if (is_rx || is_tx) {
            mem->ofi.data_trx = trx + i;
        } else {
            mem->ofi.sync_trx = trx + i;
        }

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

        // ------------------- counters
        struct fi_cntr_attr rx_cntr_attr = {
            .events = FI_CNTR_EVENTS_COMP,
            .wait_obj = FI_WAIT_UNSPEC,
        };
        if (is_rx) {
            // remote counters - count the number of fi_write/fi_read targeted to me
            m_ofi_call(fi_cntr_open(comm->domain, &rx_cntr_attr, &trx[i].rcntr, NULL));
            // uint64_t rcntr_flag = FI_REMOTE_WRITE;
            // m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].rcntr->fid, rcntr_flag));
            m_ofi_call(fi_cntr_set(trx[i].rcntr, 0));
        }
        if (is_tx) {
            // completed counter - count the number of fi_write/fi_read
            m_ofi_call(fi_cntr_open(comm->domain, &rx_cntr_attr, &trx[i].ccntr, NULL));
            uint64_t ccntr_flag = FI_WRITE | FI_READ;
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].ccntr->fid, ccntr_flag));
            m_ofi_call(fi_cntr_set(trx[i].ccntr, 0));
        }
        if (is_sync) {
            // completed counter - count the number of send
            m_ofi_call(fi_cntr_open(comm->domain, &rx_cntr_attr, &trx[i].ccntr, NULL));
            uint64_t ccntr_flag = FI_SEND | FI_WRITE;
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].ccntr->fid, ccntr_flag));
            m_ofi_call(fi_cntr_set(trx[i].ccntr, 0));
        }

        // ------------------- completion queue
        struct fi_cq_attr cq_attr = {
            // need to be able to recover the data for PSCW
            .format = OFI_CQ_FORMAT,
            .wait_obj = FI_WAIT_UNSPEC,
        };
        m_ofi_call(fi_cq_open(comm->domain, &cq_attr, &trx[i].cq, NULL));
        if (is_rx || is_tx) {
            uint64_t tcq_trx_flags = FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION;
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].cq->fid, tcq_trx_flags));
        } else if (is_sync) {
            // uint64_t tcq_trx_flags = FI_RECV;
            uint64_t tcq_trx_flags = FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION;
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].cq->fid, tcq_trx_flags));
        }

        // ------------------- finalize
        // enable the EP
        m_ofi_call(fi_enable(trx[i].ep));
        if (is_rx || is_sync) {
            // get the addresses from others
            m_rmem_call(ofi_util_av(comm->size, trx[i].ep, trx[i].av, &trx[i].addr));
        }
        if (is_rx) {
            // bind the memory registration
            if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
                uint64_t mr_trx_flags = 0;
                if (mem->ofi.mr) {
                    m_ofi_call(fi_mr_bind(mem->ofi.mr, &trx[i].ep->fid, mr_trx_flags));
                }
                m_ofi_call(fi_mr_bind(mem->ofi.signal.mr, &trx[i].ep->fid, mr_trx_flags));
            }
            // bind the remote completion counter
            if (mem->ofi.mr) {
                m_ofi_call(
                    fi_mr_bind(mem->ofi.mr, &mem->ofi.data_trx[i].rcntr->fid, FI_REMOTE_WRITE));
            }
            m_ofi_call(
                fi_mr_bind(mem->ofi.signal.mr, &mem->ofi.data_trx[i].rcntr->fid, FI_REMOTE_WRITE));
        }
        m_verb("done with EP # %d", i);
    }

    //---------------------------------------------------------------------------------------------
    // if needed, enable the MR and then get the corresponding key and share it
    // first the user region's key
    uint64_t usr_key = FI_KEY_NOTAVAIL;
    uint64_t sig_key = FI_KEY_NOTAVAIL;
    if (mem->ofi.mr) {
        if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            m_ofi_call(fi_mr_enable(mem->ofi.mr));
        }
        usr_key = fi_mr_key(mem->ofi.mr);
        m_assert(usr_key != FI_KEY_NOTAVAIL, "the key registration failed");
    }
    void* key_list = calloc(ofi_get_size(comm), sizeof(uint64_t));
    pmi_allgather(sizeof(usr_key), &usr_key, &key_list);
    mem->ofi.key_list = (uint64_t*)key_list;

    // then the signal key
    if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
        m_ofi_call(fi_mr_enable(mem->ofi.signal.mr));
    }
    sig_key = fi_mr_key(mem->ofi.signal.mr);
    m_assert(sig_key != FI_KEY_NOTAVAIL, "the key registration failed");
    key_list = calloc(ofi_get_size(comm), sizeof(uint64_t));
    pmi_allgather(sizeof(sig_key), &sig_key, &key_list);
    mem->ofi.signal.key_list = (uint64_t*)key_list;

    //---------------------------------------------------------------------------------------------
    // allocate the data user for sync
    // if needed allocate sync data structures
    mem->ofi.sync.cqdata = calloc(comm->size, sizeof(ofi_cqdata_t));
    for (int i = 0; i < comm->size; ++i) {
        ofi_cqdata_t* ccq = mem->ofi.sync.cqdata + i;
        ccq->kind = m_ofi_cq_kind_sync;
        ccq->cq = mem->ofi.sync_trx->cq;
        ccq->sync.cntr = mem->ofi.sync.epoch;
    }
    return m_success;
}
int ofi_rmem_free(ofi_rmem_t* mem, ofi_comm_t* comm) {
    struct fid_ep* nullsrx = NULL;
    struct fid_stx* nullstx = NULL;
    // free the MR
    if (mem->ofi.mr) {
        m_ofi_call(fi_close(&mem->ofi.mr->fid));
    }
    m_ofi_call(fi_close(&mem->ofi.signal.mr->fid));
    free(mem->ofi.key_list);
    free(mem->ofi.signal.key_list);

    // free the Tx first, need to close them before closing the AV in the Rx
    const int n_trx = comm->n_ctx + 1;
    for (int i = 0; i < n_trx; ++i) {
        const bool is_rx = (i < mem->ofi.n_rx);
        const bool is_tx = (i < mem->ofi.n_tx);
        const bool is_sync = (i == comm->n_ctx);

        ofi_rma_trx_t* trx = (is_sync) ? (mem->ofi.sync_trx) : (mem->ofi.data_trx + i);
        m_rmem_call(ofi_util_free_ep(comm->prov, &trx->ep, &nullstx, &nullsrx));
        if (is_tx || is_sync) {
                m_ofi_call(fi_close(&trx->ccntr->fid));
        }
        if (is_rx) {
                m_ofi_call(fi_close(&trx->rcntr->fid));
        }
        m_ofi_call(fi_close(&trx->cq->fid));
        if (is_rx || is_sync) {
            m_ofi_call(fi_close(&trx->av->fid));
            free(trx->addr);
        }
    }
    free(mem->ofi.data_trx);
    // free the sync
    free(mem->ofi.sync.icntr);
    free(mem->ofi.sync.cqdata);
    return m_success;
}

int ofi_rma_init(ofi_rma_t* rma, ofi_rmem_t* pmem, ofi_comm_t* comm) {
    // initialize the context
    rma->ofi.cq.kind = m_ofi_cq_kind_rqst;

    // local IOV
    rma->ofi.iov = (struct iovec){
        .iov_base = rma->buf,
        .iov_len = rma->count,
    };
    // remote IOV
    rma->ofi.riov = (struct fi_rma_iov){
        // offset starting from the key registration (FI_MR_SCALABLE)
        .addr = rma->disp,
        .len = rma->count,
        .key = pmem->ofi.key_list[rma->peer],
    };
    m_assert(rma->ofi.riov.key != FI_KEY_NOTAVAIL, "key must be >0");
    rma->ofi.msg = (struct fi_msg_rma){
        .msg_iov = &rma->ofi.iov,
        .desc = NULL,
        .iov_count = 1,
        .addr = 0,  // set later, once the context ID is known
        .rma_iov = &rma->ofi.riov,
        .rma_iov_count = 1,
        .data = 0,
        .context = &rma->ofi.cq.ctx,  // needed in case we do RPUT instead of PUT
                                      // (FI_SELECTIVE_COMPLETION + FI_COMPLETION)
    };
    return m_success;
}

typedef enum {
    RMA_OPT_PUT,
    RMA_OPT_RPUT,
    RMA_OPT_PUT_SIG,
} rma_opt_t;

static int ofi_rma_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm,
                           rma_opt_t op) {
    m_assert(ctx_id < comm->n_ctx, "ctx id = %d < the number of ctx = %d", ctx_id, comm->n_ctx);

    // set the completion parameters
    put->ofi.cq.cq = pmem->ofi.data_trx[ctx_id].cq;

    // address depends on the communicator context
    const int rx_id = m_get_rx(ctx_id,pmem);
    put->ofi.msg.addr =pmem->ofi.data_trx[rx_id].addr[put->peer];

    // we can use the inject (or FI_INJECT_COMPLETE) only if autoprogress cap is ON. Otherwise, not
    // reading the cq will lead to no progress, see issue https://github.com/pmodels/rmem/issues/4
    const bool auto_progress = (comm->prov->domain_attr->data_progress & FI_PROGRESS_AUTO);
    const bool do_inject = (put->count < comm->prov->tx_attr->inject_size) && auto_progress;

    // get the flags to use
    uint64_t flags =
        (do_inject ? FI_INJECT : 0x0) | (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE);
    // issue the operation
    switch (op) {
        case (RMA_OPT_PUT): {
            // no increment of the flag with PUT
            put->ofi.cq.rqst.flag = NULL;
            m_ofi_call(fi_writemsg(pmem->ofi.data_trx[ctx_id].ep, &put->ofi.msg, flags));
            m_countr_fetch_add(&pmem->ofi.sync.icntr[put->peer], 1);
        } break;
        case (RMA_OPT_RPUT): {
            // increment the flag with RPUT
            put->ofi.cq.rqst.flag = &put->ofi.completed;
            m_countr_store(put->ofi.cq.rqst.flag, 0);
            // do the communication
            m_ofi_call(
                fi_writemsg(pmem->ofi.data_trx[ctx_id].ep, &put->ofi.msg, FI_COMPLETION | flags));
            m_countr_fetch_add(&pmem->ofi.sync.icntr[put->peer], 1);
            // if inject, no CQ entry is generated, so the rput is completed upon exit
            if (do_inject) {
                m_countr_fetch_add(put->ofi.cq.rqst.flag, 1);
            }
        } break;
        case (RMA_OPT_PUT_SIG): {
            // no increment of the flag with PUT
            put->ofi.cq.rqst.flag = NULL;
            // delivery complete is needed to make sure the signal does not overlap in the network
            // with the actual data
            uint64_t sig_flags = FI_DELIVERY_COMPLETE | (do_inject ? FI_INJECT : 0x0);
            m_ofi_call(fi_writemsg(pmem->ofi.data_trx[ctx_id].ep, &put->ofi.msg, sig_flags));
            // atomic call
            sig_flags = flags | FI_FENCE;
            // must be a writemsg to use the flags, the struct msg is copied internally
            m_assert(pmem->ofi.signal.inc == 1, " the inc value must be 1");
            struct fi_ioc iov = {
                .addr = &pmem->ofi.signal.inc,
                .count = 1,
            };
            struct fi_rma_ioc rma_iov = {
                .addr = 0,
                .count = 1,
                .key = pmem->ofi.signal.key_list[put->peer],
            };
            struct fi_msg_atomic msg = {
                .msg_iov = &iov,
                .desc = NULL,
                .iov_count = 1,
                .addr = put->ofi.msg.addr,
                .rma_iov = &rma_iov,
                .rma_iov_count = 1,
                .datatype = FI_INT32,
                .op = FI_SUM,
                .data = 0,
                .context = NULL,  // FI_SELECTIVE_COMPLETION and ! FI_COMPLETION so the ctx is
                                  // ignored even if FI_CONTEXT mode is set
            };
            m_ofi_call(fi_atomicmsg(pmem->ofi.data_trx[ctx_id].ep,&msg,sig_flags));
            m_countr_fetch_add(&pmem->ofi.sync.icntr[put->peer], 2);
        } break;
    }
    return m_success;
}

int ofi_put_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_enqueue(put, pmem, ctx_id, comm, RMA_OPT_PUT);
}
int ofi_rput_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_enqueue(put, pmem, ctx_id, comm, RMA_OPT_RPUT);
}
int ofi_put_signal_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_enqueue(put, pmem, ctx_id, comm, RMA_OPT_PUT_SIG);
}

int ofi_rma_free(ofi_rma_t* rma) { return m_success; }

// notify the processes in comm of memory exposure epoch
int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // no call access in my memory can be done before the notification, it's safe to reset the
    // counters involved in the memory exposure: epoch[1:2]
    // do NOT reset the epoch[0], it's already exposed to the world!
    m_countr_store(mem->ofi.sync.epoch + 1, 0);
    m_countr_store(mem->ofi.sync.epoch + 2, 0);

    struct fi_context ctx;
    uint64_t data = m_ofi_data_set_post;
    // notify readiness to the rank list
    fi_cntr_set(mem->ofi.sync_trx->ccntr, 0);
    for (int i = 0; i < nrank; ++i) {
        // TODO: fix the context here, it's wrong to share it!
        uint64_t tag = m_ofi_tag_sync;
        m_ofi_call(fi_tsend(mem->ofi.sync_trx->ep, &data, sizeof(uint64_t), NULL,
                            mem->ofi.sync_trx->addr[rank[i]], tag, &ctx));
    }
    // wait for completion of the inject calls
    fi_cntr_wait(mem->ofi.sync_trx->ccntr, nrank, -1);
    return m_success;
}

// wait for the processes in comm to notify their exposure
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // open the handshake requests
    struct iovec iov = {
        .iov_len = sizeof(uint64_t),
    };
    struct fi_msg_tagged msg = {
        .msg_iov = &iov,
        .desc = NULL,
        .iov_count = 1,
        .tag = m_ofi_tag_sync,
        .ignore = 0x0,
        .data = 0,
    };
    for (int i = 0; i < nrank; ++i) {
        iov.iov_base = &mem->ofi.sync.cqdata[i].sync.buf;
        msg.addr = mem->ofi.sync_trx->addr[rank[i]];
        msg.context = &mem->ofi.sync.cqdata[i].ctx;
        uint64_t flags = FI_COMPLETION;
        m_ofi_call(fi_trecvmsg(mem->ofi.sync_trx->srx, &msg, flags));
    }
    while (m_countr_load(mem->ofi.sync.epoch + 0) < nrank) {
        ofi_progress(mem->ofi.sync_trx->cq);
    }
    // once we have received everybody's signal, resets epoch[0] for the next iteration
    // nobody can post until I have completed on my side, so it will no lead to data race
    m_countr_store(mem->ofi.sync.epoch + 0, 0);
    return m_success;
}

int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    fi_cntr_set(mem->ofi.sync_trx->ccntr, 0);
    // count the number of calls issued for each of the ranks and notify them
    int ttl_issued = 0;
    uint64_t tag = m_ofi_tag_sync;
    for (int i = 0; i < nrank; ++i) {
        int issued_rank = m_countr_exchange(&mem->ofi.sync.icntr[rank[i]], 0);
        ttl_issued += issued_rank;
        // notify
        ofi_rma_trx_t* trx = mem->ofi.sync_trx;
        ofi_cqdata_t* cqd = mem->ofi.sync.cqdata + i;
        cqd->sync.buf = m_ofi_data_set_cmpl | m_ofi_data_set_nops(issued_rank);
        m_ofi_call(fi_tsend(trx->ep, &cqd->sync.buf, sizeof(uint64_t), NULL, trx->addr[rank[i]],
                            tag, &cqd->ctx));
    }
    // wait for completion of the send calls, otherwise they will be delayed
    fi_cntr_wait(mem->ofi.sync_trx->ccntr, nrank, -1);
    //----------------------------------------------------------------------------------------------
    // count the number of completed calls and wait till they are all done
    // must complete all the sync call done in rmem_post (if any) + the sync call done with RMA
    uint64_t threshold = ttl_issued;
    uint64_t ttl_completed = 0;
    if (comm->n_ctx == 1) {
        fi_cntr_wait(mem->ofi.data_trx[0].ccntr, threshold, -1);
        fi_cntr_set(mem->ofi.data_trx[0].ccntr, 0);
        ttl_completed = threshold;
    } else {
        while (ttl_completed < threshold) {
            for (int i = 0; i < comm->n_ctx; ++i) {
                ofi_progress(mem->ofi.data_trx[i].cq);
                int nc = fi_cntr_read(mem->ofi.data_trx[i].ccntr);
                if (nc > 0) {
                    ttl_completed += nc;
                    m_ofi_call(fi_cntr_add(mem->ofi.data_trx[i].ccntr, (~nc + 0x1)));
                }
                // if the new value makes the value match, break
                if (ttl_completed >= ttl_issued) {
                    break;
                }
            }
        }
    }
    //----------------------------------------------------------------------------------------------
    m_assert(ttl_completed == (ttl_issued), "ttl_completed = %" PRIu64 ", ttl_issued = %d",
             ttl_completed, ttl_issued);
    return m_success;
}
int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    struct iovec iov = {
        .iov_len = sizeof(uint64_t),
    };
    struct fi_msg_tagged msg = {
        .msg_iov = &iov,
        .desc = NULL,
        .iov_count = 1,
        .tag = m_ofi_tag_sync,
        .ignore = 0x0,
        .data = 0,
    };
    for (int i = 0; i < nrank; ++i) {
        iov.iov_base = &mem->ofi.sync.cqdata[i].sync.buf;
        msg.addr = mem->ofi.sync_trx->addr[rank[i]];
        msg.context = &mem->ofi.sync.cqdata[i].ctx;
        uint64_t flags = FI_COMPLETION;
        m_ofi_call(fi_trecvmsg(mem->ofi.sync_trx->srx, &msg, flags));
    }

    //----------------------------------------------------------------------------------------------
    // compare the number of calls done to the value in the epoch if everybody has finished
    // n_rcompleted must be = the total number of calls received (including during the start sync) +
    // the sync from nrank for this sync
    if (mem->ofi.n_rx == 1) {
        while (m_countr_load(mem->ofi.sync.epoch + 1) < nrank) {
            ofi_progress(mem->ofi.sync_trx->cq);
        }
        uint64_t threshold = m_countr_load(mem->ofi.sync.epoch + 2);
        fi_cntr_wait(mem->ofi.data_trx[0].rcntr, threshold, -1);
        fi_cntr_set(mem->ofi.data_trx[0].rcntr, 0);
    } else {
        uint64_t n_rcompleted = 0;
        while (m_countr_load(mem->ofi.sync.epoch + 1) < nrank ||
               n_rcompleted < (m_countr_load(mem->ofi.sync.epoch + 2))) {
            // run progress to update the epoch counters
            ofi_progress(mem->ofi.sync_trx->cq);
            for (int i = 0; i < mem->ofi.n_rx; ++i) {
                ofi_progress(mem->ofi.data_trx[i].cq);
                // count the number of remote calls over the receive contexts
                uint64_t n_ri = fi_cntr_read(mem->ofi.data_trx[i].rcntr);
                if (n_ri > 0) {
                    n_rcompleted += n_ri;
                    // substract them form the counter as they have been taken into account must
                    // offset the remote completion counter, don't reset it to 0 as someone might
                    // already issue RMA calls to my memory as part of their post
                    m_ofi_call(fi_cntr_add(mem->ofi.data_trx[i].rcntr, (~n_ri + 0x1)));
                }
            }
        }
    }
    return m_success;
}

// end of file
