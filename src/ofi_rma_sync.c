#include "ofi.h"
#include "ofi_utils.h"
#include "ofi_rma_sync_tools.h"

//==================================================================================================
/**
 * @brief post open memory to the rank indicated in the list. This function sends the "Ready to
 * Receive" ack to the origin side
 *
 * note:
 * - if using tagged msgs for the "Down to Close" ack, the buffers are pre-posted
 */
int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("posting");
    // prepost the recv for the wait only if doing tagged msg for DTC
    if (comm->prov_mode.dtc_mode == M_OFI_DTC_TAGGED) {
        m_rmem_call(ofi_rmem_wait_fitrecv(nrank, rank, mem, comm));
    }
    m_rmem_call(ofi_rmem_post_fast(nrank, rank, mem, comm));
    m_verb("posted");
    return m_success;
}
/**
 * @brief post fast: open memory to the rank indicated in the list when using fast completion
 * mechanism (detailed in ofi_rmem_complete and ofi_rmem_wait).
 */
int ofi_rmem_post_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    // cq must be empty now unless we do AM
    if ((comm->prov_mode.rtr_mode != M_OFI_RTR_MSG) &&
        (comm->prov_mode.dtc_mode != M_OFI_DTC_MSG)) {
        m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
        m_verb("posting-fast");
    }
#endif
    // issue the post msg
    switch (comm->prov_mode.rtr_mode) {
        case (M_OFI_RTR_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RTR_ATOMIC):
            m_rmem_call(ofi_rmem_post_fiatomic(nrank, rank, mem, comm));
            break;
        case (M_OFI_RTR_MSG):
            m_rmem_call(ofi_rmem_post_fisend(nrank, rank, mem, comm));
            break;
        case (M_OFI_RTR_TAGGED):
            m_rmem_call(ofi_rmem_post_fitsend(nrank, rank, mem, comm));
            break;
    };
    //----------------------------------------------------------------------------------------------
    // wait for completion of the send calls
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
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
/**
 * @brief start the exposure epoch
 *
 * This is a blocking function, it waits for the "post" call on the target side
 */
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    m_verb("starting");
    const bool check_last =
        (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG) && (comm->prov_mode.dtc_mode != M_OFI_DTC_MSG);
    for (int i = 0; i < (mem->ofi.n_tx + check_last); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    // note: all the ctx of sync.cqdata will lead to the same epoch array
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    switch (comm->prov_mode.rtr_mode) {
        case (M_OFI_RTR_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RTR_ATOMIC):
            // wait for nrank and reset to 0 after
            m_rmem_call(ofi_util_sig_wait(&mem->ofi.sync.ps_sig, comm->rank,
                                          mem->ofi.sync_trx->addr[comm->rank],
                                          mem->ofi.sync_trx->ep, &progress, nrank));
            mem->ofi.sync.ps_sig.val = 0;
            break;
        case (M_OFI_RTR_TAGGED):
            // post the recvs if we use tagged recv
            m_rmem_call(ofi_rmem_start_fitrecv(nrank, rank, mem, comm));
            // not break!!
        case (M_OFI_RTR_MSG):
            // if we use tagged or AM: wait for completion
            // note: recv are NOT tracked by ccntr
            do {
                ofi_progress(&progress);
            } while (m_countr_load(m_rma_mepoch_post(mem)) < nrank);
            m_countr_fetch_add(m_rma_mepoch_post(mem), -nrank);
            break;
    }
    // activate progress in the helper thread
    // m_countr_store(mem->ofi.thread_arg.do_progress, 1);
    pthread_mutex_unlock(mem->ofi.thread_arg.do_progress);
#ifndef NDEBUG
    m_verb("started");
    if (check_last) {
        // the sync cq MUST be empty now
        m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    }
#endif
    return m_success;
}
int ofi_rmem_start_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // start as normal, progress has been activated
    m_rmem_call(ofi_rmem_start(nrank, rank, mem, comm));
    // we have to reset the values of icntr (set at enqueue or previous iteration)
    // we cannot do it in complete because we don't have the rank list
    for (int i = 0; i < nrank; ++i) {
        mem->ofi.sync.icntr[rank[i]] = 0;
    }
    return m_success;
}

//==================================================================================================
int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("completing");
    const bool is_order = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_ORDER);
    const bool is_fence = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
    const bool is_deliv = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL);
    rmem_complete_ack_t ack = {
        .node = {0},
        .nrank = nrank,
        .rank = rank,
        .mem = mem,
        .comm = comm,
    };
    //----------------------------------------------------------------------------------------------
    // issue the ack, if we are not fencing or using delivery complete or order, we can issue the
    // ack early if we are fencing, we have to wait for the completion of the operations enqueued in
    // the list
    int ttl_sync = 0;
    if (!is_deliv && !is_fence && !is_order) {
        ttl_sync = nrank;
        // if we are still busy in the list, we enqueue the operation, if not, we do it right away
        if (m_countr_load(&mem->ofi.qtrigr.ongoing)) {
            ack.node.kind = LNODE_KIND_COMPL;
            // the enqueue returns the device pointer but we need the host pointer
            rmem_trigr_ptr trigr_device = rmem_lmpsc_enq_val(&mem->ofi.qtrigr, &ack.node, 1);
        } else {
            m_rmem_call(ofi_rmem_issue_dtc(&ack));
        }
    }
    // just read the number of calls to wait for and reset them to 0
    int ttl_data = 0;
    for (int i = 0; i < nrank; ++i) {
        ttl_data += mem->ofi.sync.icntr[rank[i]];
    }
    int threshold = ttl_sync + ttl_data;
    m_verb("complete: waiting for %d syncs and %d calls, total %d to complete", ttl_sync, ttl_data,
           threshold);

    //----------------------------------------------------------------------------------------------
    // wait for the queue to have processed all the calls enqueued, progress is made in the async
    // thread while there still are stuff in the queue
    while (m_countr_load(&mem->ofi.qtrigr.ongoing)) {
        sched_yield();
    }
    // disable progress in the helper thread
    // m_countr_store(mem->ofi.thread_arg.do_progress, 0);
    pthread_mutex_lock(mem->ofi.thread_arg.do_progress);

    // if we are not fencing and not ordering, progress every EP now, waiting for the completion
    // if we are fencing or ordering, then progress will be made later
    if (!is_fence && !is_order) {
        m_rmem_call(ofi_rmem_progress_wait_noyield(threshold, m_rma_mepoch_local(mem),
                                                   mem->ofi.n_tx + 1, mem->ofi.data_trx,
                                                   mem->ofi.sync.epch));
    }
    //----------------------------------------------------------------------------------------------
    // send the ack if delivery complete
    if (is_deliv || is_fence || is_order) {
        // send the acknowledgement
        m_rmem_call(ofi_rmem_issue_dtc(&ack));
        m_verb("completing the delivery complete ack: %d, current = %d", nrank,
               m_countr_load(m_rma_mepoch_local(mem)));

        // make sure the ack are done, progress the sync only
        int to_wait_for = (is_fence || is_order) ? (threshold + nrank) : nrank;
        ofi_rma_trx_t* trx = (is_fence || is_order) ? mem->ofi.data_trx : mem->ofi.sync_trx;
        m_rmem_call(ofi_rmem_progress_wait_noyield(to_wait_for, m_rma_mepoch_local(mem), 1, trx,
                                                   mem->ofi.sync.epch));
    }
    //----------------------------------------------------------------------------------------------
    // reset the value of icntr, they are not needed anymore
    for (int i = 0; i < nrank; ++i) {
        mem->ofi.sync.icntr[rank[i]] = 0;
    }
    //----------------------------------------------------------------------------------------------
#ifndef NDEBUG
    // no need to check the last rx/tx if AM message is used
    const bool check_last =
        (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG) && (comm->prov_mode.dtc_mode != M_OFI_DTC_MSG);
    for (int i = 0; i < (mem->ofi.n_tx + check_last); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
    m_verb("completed");
#endif
    return m_success;
}
/**@brief complete fast: wait for threshold calls to complete on the origin side.
 *
 * note: we progress all the 
*/
int ofi_rmem_complete_fast(const int threshold, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(comm->prov_mode.rcmpl_mode != M_OFI_RCMPL_FENCE,
             "cannot complete fast with a fence mode");
    m_assert(comm->prov_mode.rcmpl_mode != M_OFI_RCMPL_ORDER,
             "cannot complete fast with a order mode");
    m_assert(comm->prov_mode.rcmpl_mode != M_OFI_RCMPL_DELIV_COMPL,
             "cannot complete fast with a delivery complete mode");
    m_verb("completing-fast: %d calls, already done: %d", threshold,
           m_countr_load(m_rma_mepoch_local(mem)));
    //----------------------------------------------------------------------------------------------
    // wait for the helper thread to be done with issuing operations
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    while (m_countr_load(&mem->ofi.qtrigr.ongoing)) {
        sched_yield();
    }
    // disable progress in the helper thread
    // m_countr_store(mem->ofi.thread_arg.do_progress, 0);
    pthread_mutex_lock(mem->ofi.thread_arg.do_progress);
    // wait for completion of the requested operations
    m_rmem_call(ofi_rmem_progress_wait_noyield(threshold, m_rma_mepoch_local(mem), mem->ofi.n_tx,
                                               mem->ofi.data_trx, mem->ofi.sync.epch));
    //----------------------------------------------------------------------------------------------
    m_verb("completed fast");
    //----------------------------------------------------------------------------------------------
    return m_success;
}

//==================================================================================================
int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("waiting");
    //----------------------------------------------------------------------------------------------
    // get the number of calls done by the origins
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    int i = 0;
    while (m_countr_load(m_rma_mepoch_cmpl(mem)) < nrank) {
        // every try progress the sync, we really need it!
        progress.cq = mem->ofi.sync_trx->cq;
        ofi_progress(&progress);

        // progress the data as well while we are at it
        // even if we do RDMA, a software ack might be needed in the case of delivery complete
        progress.cq = mem->ofi.data_trx[i].cq;
        ofi_progress(&progress);
        // update the counter to loop on the data receive trx
        i = (i + 1) % mem->ofi.n_rx;
    }
    m_countr_fetch_add(m_rma_mepoch_cmpl(mem), -nrank);
    m_verb("I have received the %d sync calls",nrank);
#ifndef NDEBUG
    // sync cq must be empty now unless we have AM posted receives
    if (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG && comm->prov_mode.dtc_mode != M_OFI_DTC_MSG) {
        m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    }
#endif
    //----------------------------------------------------------------------------------------------
    // wait for the calls to complete
    switch (comm->prov_mode.rcmpl_mode) {
        case (M_OFI_RCMPL_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RCMPL_ORDER):
        case (M_OFI_RCMPL_FENCE):
        case (M_OFI_RCMPL_DELIV_COMPL):
            // nothing to do:
            // - ordering: reception of ack notifies completion of the RDMA calls (same as fence)
            // - delivery complete: the ack is sent once completion is satisfied
            // - fence: the completion of the ack indicates completion of the RMA with the ack
            // completion semantics: see TARGET COMPLETION SEMANTICS at
            // https://ofiwg.github.io/libfabric/main/man/fi_cq.3.html
            break;
        case (M_OFI_RCMPL_REMOTE_CNTR): {
            // zero the remote counter and wait for completion
            // when not using remote counters, every put coming will add 1 to the value of
            // remote(mem), so the value will always be negative here.
            int threshold = -1 * m_countr_exchange(m_rma_mepoch_remote(mem), 0);
            // the counter is linked to the MR so waiting on it will trigger progress
            m_assert(threshold >= 0, "the threshold = %d must be >=0", threshold);
            m_ofi_call(fi_cntr_wait(mem->ofi.rcntr, threshold, -1));
            m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
        } break;
        case (M_OFI_RCMPL_CQ_DATA): {
            // WARNING: every put comes with data that will add 1 to the epoch[2] value
            // wait for it to go back up to 0
            m_rmem_call(ofi_rmem_progress_wait_noyield(0, m_rma_mepoch_remote(mem), mem->ofi.n_tx,
                                                       mem->ofi.data_trx, mem->ofi.sync.epch));
        } break;
    };
    //----------------------------------------------------------------------------------------------
#ifndef NDEBUG
    m_verb("waited-all");
    m_assert(m_countr_load(m_rma_mepoch_cmpl(mem)) == 0, "ohoh value is not 0: %d",
             m_countr_load(m_rma_mepoch_cmpl(mem)));
    m_assert(m_countr_load(m_rma_mepoch_remote(mem)) == 0, "ohoh value is not 0: %d",
             m_countr_load(m_rma_mepoch_remote(mem)));
    m_verb("waited");
    // no need to check the last rx/tx if we do AM
    for (int j = 0; j < (mem->ofi.n_tx + (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG)); ++j) {
        m_mem_check_empty_cq(mem->ofi.data_trx[j].cq);
    }
#endif
    return m_success;
}

int ofi_rmem_wait_fast(const int ncalls, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(comm->prov_mode.rcmpl_mode != M_OFI_RCMPL_FENCE,
             "cannot complete fast with a fence mode");
    m_assert(comm->prov_mode.rcmpl_mode != M_OFI_RCMPL_DELIV_COMPL,
             "cannot complete fast with a delivery complete mode");
    m_verb("waiting-fast: %d calls, already done: %d", ncalls,
           m_countr_load(m_rma_mepoch_local(mem)));
    m_verb("waiting fast");
    m_verb("wait untill: waiting for %d calls to complete", ncalls);
    switch (comm->prov_mode.rcmpl_mode) {
        case (M_OFI_RCMPL_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RCMPL_ORDER):
        case (M_OFI_RCMPL_FENCE):
        case (M_OFI_RCMPL_DELIV_COMPL):
            m_assert(ncalls == 0,
                     "CANNOT do fast completion mechanism with non-zero (%d) calls to wait when "
                     "using delivery complete",
                     ncalls);
            break;
        case (M_OFI_RCMPL_REMOTE_CNTR): {
            // the counter is linked to the MR so waiting on it will trigger progress
            m_assert(ncalls >=0,"the threshold = %d must be >=0",ncalls);
            m_ofi_call(fi_cntr_wait(mem->ofi.rcntr, ncalls, -1));
            m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
        } break;
        case (M_OFI_RCMPL_CQ_DATA): {
            // WARNING: every put comes with data that will add 1 to the epoch[2] value
            // to make sure the value is always bellow the threshold (0 here), we need to substract
            // the number of calls to wait for
            m_countr_fetch_add(m_rma_mepoch_remote(mem), -ncalls);
            // wait for it to come down
            m_rmem_call(ofi_rmem_progress_wait_noyield(0, m_rma_mepoch_remote(mem), mem->ofi.n_tx,
                                               mem->ofi.data_trx, mem->ofi.sync.epch));
        } break;
    }
    m_verb("waited fast");
    return m_success;
}

// end of file
