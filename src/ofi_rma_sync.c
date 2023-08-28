#include "ofi.h"
#include "ofi_utils.h"
#include "ofi_rma_sync_tools.h"

//==================================================================================================
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
// notify the processes in comm of memory exposure epoch
int ofi_rmem_post_fast(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    // cq must be empty now unless we do AM
    if (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG) {
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
// wait for the processes in comm to notify their exposure
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
#ifndef NDEBUG
    m_verb("starting");
    for (int i = 0; i < (mem->ofi.n_tx + (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG)); ++i) {
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
#ifndef NDEBUG
    m_verb("started");
    if (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG) {
        // the sync cq MUST be empty now
        m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    }
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
    int ttl_data = 0;
    int ttl_sync = 0;
    if (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL) {
        // just read the number of calls to wait for and reset them to 0
        // the other side has to wait for nothing
        for (int i = 0; i < nrank; ++i) {
            int issued_rank = m_countr_exchange(&mem->ofi.sync.icntr[rank[i]], 0);
            ttl_data += issued_rank;
        }
        ttl_sync = 0;
    } else {
        // send the ack if not delivery complete
        ttl_sync = nrank;
        switch (comm->prov_mode.dtc_mode) {
            case (M_OFI_DTC_NULL):
                m_assert(0, "should not be NULL here");
                break;
            case (M_OFI_DTC_TAGGED):
                m_rmem_call(ofi_rmem_complete_fitsend(nrank, rank, mem, comm, &ttl_data));
                break;
            case (M_OFI_DTC_MSG):
                m_rmem_call(ofi_rmem_complete_fisend(nrank, rank, mem, comm, &ttl_data));
                break;
        };
    }

    // get the correct threshold value depending if we have a signal or not
    uint64_t threshold;
    switch (comm->prov_mode.sig_mode) {
        case (M_OFI_SIG_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_SIG_ATOMIC): {
            threshold = ttl_sync + ttl_data + m_countr_exchange(&mem->ofi.sync.isig, 0);
            m_verb("complete: waiting for %d syncs, %d calls and %d signals to complete", ttl_sync,
                   ttl_data, m_countr_load(&mem->ofi.sync.isig));
        } break;
        case (M_OFI_SIG_CQ_DATA): {
            threshold = ttl_sync + ttl_data;
            m_verb("complete: waiting for %d syncs, %d calls (total: %llu)", ttl_sync, ttl_data,
                   threshold);

        } break;
    }
    // use complete fast to what for the threshold
    m_rmem_call(ofi_rmem_complete_fast(threshold, mem, comm));

    // send the ack if delivery complete
    if (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL) {
        // as we have already used the values in icntr, it's gonna be 0.
        // this is expected as there is no calls to wait for on the target side
        switch (comm->prov_mode.dtc_mode) {
            case (M_OFI_DTC_NULL):
                m_assert(0, "should not be NULL here");
                break;
            case (M_OFI_DTC_TAGGED):
                m_rmem_call(ofi_rmem_complete_fitsend(nrank, rank, mem, comm, &ttl_data));
                break;
            case (M_OFI_DTC_MSG):
                m_rmem_call(ofi_rmem_complete_fisend(nrank, rank, mem, comm, &ttl_data));
                break;
        };
        // make sure the ack are done
        m_rmem_call(ofi_rmem_complete_fast(nrank, mem, comm));
    }
    return m_success;
}
int ofi_rmem_complete_fast(const int threshold, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_verb("completing-fast: %d calls, already done: %d", threshold,
           m_countr_load(m_rma_mepoch_local(mem)));
    //----------------------------------------------------------------------------------------------
    // rma calls generate cq entries so they need to be processed, we loop on the data_trx only
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
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
    // no need to check the last rx/tx if AM message is used
    for (int i = 0; i < (mem->ofi.n_tx + (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG)); ++i) {
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
#ifndef NDEBUG
    // sync cq must be empty now unless we have AM posted receives
    if (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG) {
        m_mem_check_empty_cq(mem->ofi.sync_trx->cq);
    }
#endif
    // TODO: this is not optimal as we add the threshold back to the atomic if needed in wait_fast
    uint64_t threshold = m_countr_exchange(m_rma_mepoch_remote(mem), 0);
    //----------------------------------------------------------------------------------------------
    // wait for the calls to complete
    m_verb("waitall: waiting for %llu calls to complete", threshold);
    ofi_rmem_wait_fast(threshold, mem, comm);

#ifndef NDEBUG
    m_assert(m_countr_load(m_rma_mepoch_cmpl(mem)) == 0, "ohoh");
    m_assert(m_countr_load(m_rma_mepoch_remote(mem)) == 0, "ohoh");
    m_verb("waited");
    // no need to check the last rx/tx if we do AM
    for (int i = 0; i < (mem->ofi.n_tx + (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG)); ++i) {
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
        case (M_OFI_RCMPL_DELIV_COMPL):
            m_assert(ncalls == 0,
                     "CANNOT do fast completion mechanism with non-zero (%d) calls to wait",
                     ncalls);
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
                .xctx.epoch_ptr = mem->ofi.sync.epch,
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
    // no need to check the last rx/tx if we do AM
    for (int i = 0; i < (mem->ofi.n_tx + (comm->prov_mode.rtr_mode != M_OFI_RTR_MSG)); ++i) {
        m_mem_check_empty_cq(mem->ofi.data_trx[i].cq);
    }
#endif
    return m_success;
}

// end of file
