#include "ofi.h"
#include "ofi_utils.h"

//==================================================================================================
#ifndef NDEBUG
#define m_mem_check_empty_cq(cq)                                                    \
    do {                                                                            \
        ofi_cq_entry event[1];                                                      \
        int ret = fi_cq_read(cq, event, 1);                                         \
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
    { ((void)0); }

#endif

// post the receive for the Start step
static int ofi_rmem_start_firecv(const int nrank, const int* rank, ofi_rmem_t* mem,
                                 ofi_comm_t* comm) {
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
        .fallback_ctx = &mem->ofi.sync.cqdata_ps->ctx,
    };
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps + i;
        cqdata->kind = m_ofi_cq_kind_sync;
        iov.iov_base = &cqdata->sync.data;
        msg.desc = &cqdata->sync.mr.desc;
        msg.context = &cqdata->ctx;
        msg.addr = mem->ofi.sync_trx->addr[rank[i]];
        uint64_t flags = FI_COMPLETION;
        m_ofi_call_again(fi_trecvmsg(mem->ofi.sync_trx->srx, &msg, flags), &progress);
    }
    return m_success;
}

static int ofi_rmem_post_fisend(const int nrank, const int* rank, ofi_rmem_t* mem,
                                ofi_comm_t* comm) {
    // notify readiness to the rank list
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_ps->ctx,
    };
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps + i;
        cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqdata->sync.data = m_ofi_data_set_post;
        uint64_t tag = m_ofi_tag_set_ps;
        m_ofi_call_again(
            fi_tsend(mem->ofi.sync_trx->ep, &cqdata->sync.data, sizeof(uint64_t),
                     cqdata->sync.mr.desc, mem->ofi.sync_trx->addr[rank[i]], tag, &cqdata->ctx),
            &progress);
    }
    return m_success;
}

static int ofi_rmem_post_fiatomic(const int nrank, const int* rank, ofi_rmem_t* mem,
                                  ofi_comm_t* comm) {
    // notify readiness to the rank list, posting uses cqdata_ps to complete
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_ps->ctx,
    };
    mem->ofi.sync.rtr.inc = 1;
    for (int i = 0; i < nrank; ++i) {
        // used for completion
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps + i;
        cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        // issue the atomic, local var
        struct fi_ioc iov = {
            .count = 1,  // depends on datatype
            .addr = &mem->ofi.sync.rtr.inc,
        };
        struct fi_rma_ioc rma_iov = {
            .count = 1,  // depends on datatype
            .addr = mem->ofi.sync.rtr.val_mr.base_list[rank[i]] + 0,
            .key = mem->ofi.sync.rtr.val_mr.key_list[rank[i]],
        };
        struct fi_msg_atomic msg = {
            .msg_iov = &iov,
            .desc = &mem->ofi.sync.rtr.inc_mr.desc,
            .iov_count = 1,
            .addr = mem->ofi.sync_trx->addr[rank[i]],
            .rma_iov = &rma_iov,
            .rma_iov_count = 1,
            .datatype = FI_INT32,
            .op = FI_SUM,
            .data = 0x0,  // atomics does NOT support FI_REMOTE_CQ_DATA
            .context = &cqdata->ctx,
        };

        m_verb("atomic: of size %lu and %lu", iov.count, rma_iov.count);
        m_verb("atomic: addr = %p", iov.addr);
        m_verb("atomic: inc = %d", mem->ofi.sync.rtr.inc);
        m_ofi_call_again(fi_atomicmsg(mem->ofi.sync_trx->ep, &msg, FI_TRANSMIT_COMPLETE),
                         &progress);
    }
    return m_success;
}
static int ofi_rmem_start_fiatomic(const int nrank, const int* rank, ofi_rmem_t* mem,
                                   ofi_comm_t* comm) {
    // notify readiness to the rank list
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_ps->ctx,
    };
    // used for completion, only the first one
    ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps;
    cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
    // issue the atomic
    int myself = comm->rank;
    // even we read it, we need to provide a source buffer
    struct fi_ioc iov = {
        .addr = &mem->ofi.sync.rtr.inc,
        .count = 1,
    };
    struct fi_ioc res_iov = {
        .addr = &mem->ofi.sync.rtr.res,
        .count = 1,
    };
    struct fi_rma_ioc rma_iov = {
        .count = 1,
        .addr = mem->ofi.sync.rtr.val_mr.base_list[myself] + 0,
        .key = mem->ofi.sync.rtr.val_mr.key_list[myself],
    };
    struct fi_msg_atomic msg = {
        .msg_iov = &iov,
        .desc = &mem->ofi.sync.rtr.inc_mr.desc,
        .iov_count = 1,                           // 1,
        .addr = mem->ofi.sync_trx->addr[myself],  // myself
        .rma_iov = &rma_iov,
        .rma_iov_count = 1,
        .datatype = FI_INT32,
        .op = FI_ATOMIC_READ,
        .data = 0x0,  // atomics does NOT support FI_REMOTE_CQ_DATA
        .context = &cqdata->ctx,
    };
    m_verb("atomic: of size %lu and %lu", iov.count, rma_iov.count);
    m_ofi_call_again(fi_fetch_atomicmsg(mem->ofi.sync_trx->ep, &msg, &res_iov,
                                        &mem->ofi.sync.rtr.res_mr.desc, 1, FI_TRANSMIT_COMPLETE),
                     &progress);
    return m_success;
}

static int ofi_rmem_complete_fisend(const int nrank, const int* rank, ofi_rmem_t* mem,
                                    ofi_comm_t* comm, int* ttl_data) {
    // count the number of calls issued for each of the ranks and notify them
    *ttl_data = 0;
    uint64_t tag = m_ofi_tag_set_cw;
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_cw->ctx,
    };
    for (int i = 0; i < nrank; ++i) {
        int issued_rank = m_countr_exchange(&mem->ofi.sync.icntr[rank[i]], 0);
        *ttl_data += issued_rank;
        // notify
        ofi_rma_trx_t* trx = mem->ofi.sync_trx;
        ofi_cqdata_t* cqd = mem->ofi.sync.cqdata_cw + i;
        cqd->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqd->sync.data = m_ofi_data_set_cmpl | m_ofi_data_set_nops(issued_rank);
        m_verb("complete_fisend: I have done %d write to %d, value sent = %llu",issued_rank,i,cqd->sync.data);
        m_ofi_call_again(fi_tsend(trx->ep, &cqd->sync.data, sizeof(uint64_t), cqd->sync.mr.desc,
                                  trx->addr[rank[i]], tag, &cqd->ctx),
                         &progress);
    }
    return m_success;
}

static int ofi_rmem_wait_firecv(const int nrank, const int* rank, ofi_rmem_t* mem,
                                ofi_comm_t* comm) {
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
        .fallback_ctx = &mem->ofi.sync.cqdata_cw->ctx,
    };
    uint64_t flags = FI_COMPLETION;
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_cw + i;
        cqdata->kind = m_ofi_cq_kind_sync;
        iov.iov_base = &cqdata->sync.data;
        msg.desc = &cqdata->sync.mr.desc;
        msg.context = &cqdata->ctx;
        msg.addr = mem->ofi.sync_trx->addr[rank[i]];
        m_ofi_call_again(fi_trecvmsg(mem->ofi.sync_trx->srx, &msg, flags), &progress);
    }
    return m_success;
}
//==================================================================================================
int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("posting");
    // prepost the recv for the wait
    m_rmem_call(ofi_rmem_wait_firecv(nrank, rank, mem, comm));
    m_rmem_call(ofi_rmem_post_fast(nrank, rank, mem, comm));
    m_verb("posted");
    return m_success;
}
// notify the processes in comm of memory exposure epoch
int ofi_rmem_post_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    // cq must be empty now
    m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    m_verb("posting-fast");
#endif
    // issue the post msg
    switch (comm->prov_mode.rtr_mode) {
        case (M_OFI_RTR_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RTR_ATOMIC):
            m_rmem_call(ofi_rmem_post_fiatomic(nrank, rank, mem, comm));
            break;
        case (M_OFI_RTR_TMSG):
            m_rmem_call(ofi_rmem_post_fisend(nrank, rank, mem, comm));
            break;
    };
    //----------------------------------------------------------------------------------------------
    // wait for completion of the send calls
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_ps->ctx,
    };
    while (m_countr_load(m_rma_mepoch_local(mem)) < nrank) {
        m_rmem_call(ofi_progress(&progress));
    }
    m_countr_fetch_add(m_rma_mepoch_local(mem), -nrank);

    //----------------------------------------------------------------------------------------------
#ifndef NDEBUG
    // cq must be empty now -> NOT true because we prepost the recvs
    // m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    m_verb("posted-fast");
#endif
    return m_success;
}
//==================================================================================================
// wait for the processes in comm to notify their exposure
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    m_verb("starting");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif

    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_ps->ctx,
    };
    switch (comm->prov_mode.rtr_mode) {
        case (M_OFI_RTR_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RTR_ATOMIC): {
            // wait for nrank and reset to 0 after
            m_rmem_call(ofi_util_sig_wait(&mem->ofi.sync.rtr, comm->rank,
                                               mem->ofi.sync_trx->addr[comm->rank],
                                               mem->ofi.sync_trx->ep, &progress, nrank));
            mem->ofi.sync.rtr.val = 0;
            // read the current value
            // int it = 0;
            // int cntr_cur = m_countr_load(m_rma_mepoch_local(mem));
            // while (mem->ofi.sync.rtr.res < nrank) {
            //     // issue a fi_fetch
            //     m_verb("issuing an atomic number %d, res = %d", it, mem->ofi.sync.rtr.res);
            //     m_rmem_call(ofi_rmem_start_fiatomic(nrank, rank, mem, comm));
            //     // count the number of issued atomics
            //     it++;
            //     // wait for completion of the atomic
            //     while (m_countr_load(m_rma_mepoch_local(mem)) < (cntr_cur + it)) {
            //         ofi_progress(&progress);
            //     }
            //     m_verb("atomics has completed, res = %d", mem->ofi.sync.rtr.res);
            // }
            // m_countr_fetch_add(m_rma_mepoch_local(mem), -it);
            // reset for next time
        } break;
        case (M_OFI_RTR_TMSG): {
            // post the recvs
            m_rmem_call(ofi_rmem_start_firecv(nrank, rank, mem, comm));
            // wait for completion, recv are NOT tracked by ccntr
            // all the ctx of sync.cqdata will lead to the same epoch array
            do {
                ofi_progress(&progress);
            } while (m_countr_load(m_rma_mepoch_post(mem)) < nrank);
            m_countr_fetch_add(m_rma_mepoch_post(mem), -nrank);
        } break;
    }
#ifndef NDEBUG
    // the sync cq MUST be empty now
    m_verb("started");
    m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
#endif
    return m_success;
}
int ofi_rmem_start_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_rmem_call(ofi_rmem_start(nrank, rank, mem, comm));
    // reset the value of icntr, it's needed if we use the fast completion mechanism
    for (int i = 0; i < nrank; ++i) {
        m_countr_exchange(&mem->ofi.sync.icntr[rank[i]], 0);
    }

    return m_success;
}

//==================================================================================================
int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("completing");
    // send the ack
    int ttl_data;
    m_rmem_call(ofi_rmem_complete_fisend(nrank, rank, mem, comm, &ttl_data));

    // get the correct threshold value
    uint64_t threshold;
    switch (comm->prov_mode.sig_mode) {
        case (M_OFI_SIG_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_SIG_ATOMIC): {
            threshold = nrank + ttl_data + m_countr_exchange(&mem->ofi.sync.isig, 0);
            m_verb("complete: waiting for %d syncs, %d calls and %d signals to complete", nrank,
                   ttl_data, m_countr_load(&mem->ofi.sync.isig));
        } break;
        case (M_OFI_SIG_CQ_DATA): {
            threshold = nrank + ttl_data;
            m_verb("complete: waiting for %d syncs, %d calls (total: %llu)", nrank, ttl_data,
                   threshold);

        } break;
    }
    // use complete fast to what for the threshold
    m_rmem_call(ofi_rmem_complete_fast(threshold, mem, comm));
    return m_success;
}
int ofi_rmem_complete_fast(const int threshold, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("completing-fast: %d calls", threshold);
    //----------------------------------------------------------------------------------------------
    // rma calls generate cq entries so they need to be processed, we loop on the data_trx only
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_cw->ctx,
    };
    int i = 0;
    // we have to complete on all the trx, sync and data
    while (m_countr_load(m_rma_mepoch_local(mem)) < threshold) {
        progress.cq = mem->ofi.data_trx[i].cq;
        m_ofi_call(ofi_progress(&progress));
        i = (i + 1) % (mem->ofi.n_tx + 1);
    }
    m_countr_fetch_add(m_rma_mepoch_local(mem), -threshold);

#ifndef NDEBUG
    m_verb("completed-fast");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

//==================================================================================================
int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("waiting");
    //----------------------------------------------------------------------------------------------
    // get the number of calls done by the origins
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .fallback_ctx = &mem->ofi.sync.cqdata_cw->ctx,
    };
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
    m_countr_fetch_add(m_rma_mepoch_cmpl(mem), -nrank);
    // TODO: this is not optimal as we add the threshold back to the atomic if needed in wait_fast
    uint64_t threshold = m_countr_exchange(m_rma_mepoch_remote(mem), 0);
    m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    //----------------------------------------------------------------------------------------------
    // wait for the calls to complete
    m_verb("waitall: waiting for %llu calls to complete", threshold);
    ofi_rmem_wait_fast(threshold, mem, comm);

#ifndef NDEBUG
    m_assert(m_countr_load(m_rma_mepoch_cmpl(mem)) == 0, "ohoh");
    m_assert(m_countr_load(m_rma_mepoch_remote(mem)) == 0, "ohoh");
    m_verb("waited");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

int ofi_rmem_wait_fast(const int ncalls, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("waiting fast");
    m_verb("wait untill: waiting for %d calls to complete", ncalls);
    switch (comm->prov_mode.rcmpl_mode) {
        case (M_OFI_RCMPL_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RCMPL_REMOTE_CNTR): {
            // the counter is linked to the MR so waiting on it will trigger progress
            m_ofi_call(fi_cntr_wait(mem->ofi.rcntr, ncalls, -1));
            m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
        } break;
        case (M_OFI_RCMPL_CQ_DATA): {
            // every put comes with data that will substract 1 to the epoch[2] value
            // first bump the value of epoch[2]
            m_countr_fetch_add(m_rma_mepoch_remote(mem), ncalls);
            // wait for it to come down
            ofi_progress_t progress = {
                .cq = mem->ofi.sync_trx->cq,
                .fallback_ctx = &mem->ofi.sync.cqdata_cw->ctx,
            };
            int i = 0;
            while (m_countr_load(m_rma_mepoch_remote(mem)) > 0) {
                progress.cq = mem->ofi.data_trx[i].cq;
                ofi_progress(&progress);
                // update the counter
                i = (i + 1) % mem->ofi.n_tx;
            }
        } break;
        case (M_OFI_RCMPL_FENCE): {
            m_assert(0, "idk what to do");
        } break;
    }
#ifndef NDEBUG
    m_verb("waited fast");
    for (int i = 0; i < (mem->ofi.n_tx + 1); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}
