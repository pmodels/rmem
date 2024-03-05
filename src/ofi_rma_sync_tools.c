#include "ofi_rma_sync_tools.h"
#include "ofi_utils.h"
#include "rdma/fabric.h"

//==================================================================================================
// COMMON TAGGEG and AM MSG
//==================================================================================================
typedef enum {
    M_RMEM_ACK_TAGGED,
    M_RMEM_ACK_AM,
} rmem_ack_t;
static int ofi_rmem_post_send(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm,
                              rmem_ack_t ack) {
    // notify readiness to the rank list
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps + i;
        cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqdata->sync.data = m_ofi_data_set_post;
        m_verb("cqdata ctx %p, kind = local %d, epoch_ptr = %p", &cqdata->ctx,
               cqdata->kind & m_ofi_cq_inc_local, cqdata->epoch_ptr);
        switch (ack) {
            case (M_RMEM_ACK_AM): {
                m_ofi_call_again(
                    fi_send(mem->ofi.sync_trx->ep, &cqdata->sync.data, sizeof(uint64_t),
                            cqdata->sync.mr.desc, mem->ofi.sync_trx->addr[rank[i]], &cqdata->ctx),
                    &progress);
            } break;
            case (M_RMEM_ACK_TAGGED): {
                uint64_t tag = m_ofi_tag_set_ps;
                m_ofi_call_again(fi_tsend(mem->ofi.sync_trx->ep, &cqdata->sync.data,
                                          sizeof(uint64_t), cqdata->sync.mr.desc,
                                          mem->ofi.sync_trx->addr[rank[i]], tag, &cqdata->ctx),
                                 &progress);
            } break;
        }
    }
    return m_success;
}
static int ofi_rmem_complete_send(const int nrank, const int* rank, ofi_rmem_t* mem,
                                  ofi_comm_t* comm, rmem_ack_t ack) {
    // get the remote completion mode
    const bool is_fence = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
    const bool is_dcmpl = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL);
    // count the number of calls issued for each of the ranks and notify them
    const uint64_t flag = FI_TRANSMIT_COMPLETE | ((is_fence) ? FI_FENCE : 0x0);
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    for (int i = 0; i < nrank; ++i) {
        int issued_rank = 0;
        // if we are delivery complete, this is no needed
        if (!is_dcmpl && !is_fence) {
            issued_rank = mem->ofi.sync.icntr[rank[i]];
        }

        // if using fence, we need to fence ALL of the data_trx, if not we only issue on the sync
        // to avoid duplication of all the resources we forbid the user to use more than one data
        // trx when using the fence
        m_assert(!(is_fence && mem->ofi.n_tx > 1), "you cannot fence with more than 1 data TRX");
        // we cannot use fi_msg for the fence because the buffers have been posted on the sync_trx
        // and not on the data ones.
        m_assert(!(is_fence && comm->prov_mode.dtc_mode == M_OFI_DTC_MSG),
                 "you cannot fence when using MSG as a down-to-close (DTC) mode. This because the "
                 "AM buffers have been posted on the sync_trx EP, while the fence is posted on the "
                 "data_trx EP");
        ofi_rma_trx_t* trx = (is_fence) ? mem->ofi.data_trx : mem->ofi.sync_trx;

        // notify
        ofi_cqdata_t* cqd = mem->ofi.sync.cqdata_cw + i;
        cqd->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqd->sync.data = m_ofi_data_set_cmpl | m_ofi_data_set_nops(issued_rank);
        cqd->epoch_ptr = mem->ofi.sync.epch;
        m_verb("cqdata ctx %p, kind = local %d, epoch_ptr = %p", &cqd->ctx,
               cqd->kind & m_ofi_cq_inc_local, cqd->epoch_ptr);
        m_verb("complete_send: I have done %d write to %d, value sent = %llu", issued_rank, i,
               cqd->sync.data);
        struct iovec iov = {
            .iov_base = &cqd->sync.data,
            .iov_len = sizeof(uint64_t),
        };
        switch (ack) {
            case (M_RMEM_ACK_AM): {
                struct fi_msg msg = {
                    .msg_iov = &iov,
                    .desc = &cqd->sync.mr.desc,
                    .iov_count = 1,
                    .addr = trx->addr[rank[i]],
                    .context = &cqd->ctx,
                    .data = 0x0,
                };
                m_verb("complete using fi_sendmsg with EP %p",trx->ep);
                m_ofi_call_again(fi_sendmsg(trx->ep, &msg, flag), &progress);

            } break;
            case (M_RMEM_ACK_TAGGED): {
                const uint64_t tag = m_ofi_tag_set_cw;
                struct fi_msg_tagged msg = {
                    .msg_iov = &iov,
                    .desc = &cqd->sync.mr.desc,
                    .iov_count = 1,
                    .addr = trx->addr[rank[i]],
                    .tag = tag,
                    .ignore = 0x0,
                    .context = &cqd->ctx,
                    .data = 0x0,
                };
                m_verb("complete using fi_tsendmsg on EP %p, FI_FENCE? %d",trx->ep,(flag&FI_FENCE)>0);
                m_ofi_call_again(fi_tsendmsg(trx->ep, &msg, flag), &progress);
            } break;
        }
    }
    return m_success;
}

//==================================================================================================
// AM 
//==================================================================================================
int ofi_rmem_am_repost(ofi_cqdata_t* cqdata, ofi_progress_t* progress) {
    m_assert(cqdata->kind == m_ofi_cq_kind_am, "wrong type of cq data");
    ofi_am_buf_t* am = cqdata->am.buf;
    struct iovec iov = {
        .iov_len = m_ofi_am_buf_size,
        .iov_base = am->buf,
    };
    m_assert(iov.iov_len >= m_ofi_am_max_size,
             "the len = %lu should be larger than the max AM payload = %d", iov.iov_len,
             m_ofi_am_max_size);
    struct fi_msg msg = {
        .msg_iov = &iov,
        .desc = &am->mr.desc,
        .iov_count = 1,
        .addr = FI_ADDR_UNSPEC,
        .context = &cqdata->ctx,
        .data = 0,
    };
    uint64_t flags = FI_COMPLETION | FI_MULTI_RECV;
    m_ofi_call_again(fi_recvmsg(cqdata->am.srx, &msg, flags), progress);
    return m_success;
}

int ofi_rmem_am_init(ofi_rmem_t* mem, ofi_comm_t* comm) {
    // set the min buffer unavailable size
    size_t optlen = m_ofi_am_cq_min_size;
    m_ofi_call(fi_setopt(&mem->ofi.sync_trx->srx->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
                         &optlen, sizeof(optlen)));
    // allocate the am resources
    mem->ofi.sync.am.buf = m_malloc(m_ofi_am_buf_num * sizeof(ofi_am_buf_t));
    mem->ofi.sync.am.cqdata = m_malloc(m_ofi_am_buf_num * sizeof(ofi_cqdata_t));
    // post the buf to the EP
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    for (int i = 0; i < m_ofi_am_buf_num; ++i) {
        // register the am memory
        ofi_am_buf_t* am_buf = mem->ofi.sync.am.buf + i;
        am_buf->buf = m_malloc(m_ofi_am_buf_size);
        m_rmem_call(ofi_util_mr_reg(am_buf->buf, m_ofi_am_buf_size, FI_RECV, comm, &am_buf->mr.mr,
                                    &am_buf->mr.desc, NULL));
        m_rmem_call(ofi_util_mr_bind(mem->ofi.sync_trx->ep, am_buf->mr.mr, NULL, comm));
        m_rmem_call(ofi_util_mr_enable(am_buf->mr.mr, comm, NULL));
        // populate the cqdata
        m_verb("creating AM buffer %p (size %d), posting to srx %p", am_buf->buf, m_ofi_am_buf_size,
               mem->ofi.sync_trx->srx);
        ofi_cqdata_t* cqdata = mem->ofi.sync.am.cqdata + i;
        cqdata->kind = m_ofi_cq_kind_am;
        cqdata->am.buf = am_buf;
        cqdata->am.srx = mem->ofi.sync_trx->srx;
        cqdata->epoch_ptr = mem->ofi.sync.epch;
        // post the buffers now
        ofi_rmem_am_repost(cqdata, &progress);
    }
    return m_success;
}

int ofi_rmem_am_free(ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(mem->ofi.sync.cqdata_ps, "cqdata_ps must be allocated here");
    m_verb("canceling the requests");
    for (int i = 0; i < m_ofi_am_buf_num; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.am.cqdata + i;
        m_ofi_call(fi_cancel(&mem->ofi.sync_trx->srx->fid, &cqdata->ctx));
        ofi_am_buf_t* am_buf = mem->ofi.sync.am.buf + i;
        m_rmem_call(ofi_util_mr_close(am_buf->mr.mr));
        free(am_buf->buf);
    }
    free(mem->ofi.sync.am.cqdata);
    free(mem->ofi.sync.am.buf);
    return m_success;
}
int ofi_rmem_post_fisend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    return ofi_rmem_post_send(nrank, rank, mem, comm, M_RMEM_ACK_AM);
}
int ofi_rmem_complete_fisend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(comm->prov_mode.rcmpl_mode != M_OFI_RCMPL_FENCE,
             "we cannot used MSG to close a FENCE completion tracking mode");
    return ofi_rmem_complete_send(nrank, rank, mem, comm, M_RMEM_ACK_AM);
}

//==================================================================================================
// TAGGED MSG
//==================================================================================================
int ofi_rmem_post_fitsend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    return ofi_rmem_post_send(nrank, rank, mem, comm, M_RMEM_ACK_TAGGED);
}
int ofi_rmem_complete_fitsend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    return ofi_rmem_complete_send(nrank, rank, mem, comm, M_RMEM_ACK_TAGGED);
}
int ofi_rmem_start_fitrecv(const int nrank, const int* rank, ofi_rmem_t* mem,
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
        .xctx.epoch_ptr = mem->ofi.sync.epch,
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


int ofi_rmem_wait_fitrecv(const int nrank, const int* rank, ofi_rmem_t* mem,
                                ofi_comm_t* comm) {
    const bool is_fence = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
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
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    uint64_t flags = FI_COMPLETION;

    // if we use the fence, we need to close ALL the data_trx
    m_assert(!(is_fence && mem->ofi.n_tx > 1), "you cannot fence with more than 1 data TRX");
    m_assert(!(is_fence && comm->prov_mode.dtc_mode == M_OFI_DTC_MSG),
             "you cannot fence when using MSG as a down-to-close (DTC) mode");
    ofi_rma_trx_t* trx = (is_fence) ? mem->ofi.data_trx : mem->ofi.sync_trx;
    for (int i = 0; i < nrank; ++i) {
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_cw + i;
        cqdata->kind = m_ofi_cq_kind_sync;
        cqdata->epoch_ptr = mem->ofi.sync.epch;
        iov.iov_base = &cqdata->sync.data;
        msg.desc = &cqdata->sync.mr.desc;
        msg.context = &cqdata->ctx;
        msg.addr = trx->addr[rank[i]];
        m_ofi_call_again(fi_trecvmsg(trx->srx, &msg, flags), &progress);
    }
    return m_success;
}

//==================================================================================================
// RMA with CQ_DATA
//==================================================================================================
int ofi_rmem_complete_fiwrite(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // get the remote completion mode
    // count the number of calls issued for each of the ranks and notify them
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    for (int i = 0; i < nrank; ++i) {
        // if using fence, we need to fence ALL of the data_trx, if not we only issue on the sync
        // to avoid duplication of all the resources we forbid the user to use more than one data
        // trx when using the fence
        m_assert(mem->ofi.n_tx == 1, "you cannot rely on ordering with more than 1 data TRX");
        ofi_rma_trx_t* trx = mem->ofi.data_trx;

        // notify
        ofi_cqdata_t* cqd = mem->ofi.sync.cqdata_cw + i;
        cqd->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        cqd->sync.data = 0;
        cqd->epoch_ptr = mem->ofi.sync.epch;
        // get the cq data
        m_verb("cqdata ctx %p, kind = local %d, epoch_ptr = %p", &cqd->ctx,
               cqd->kind & m_ofi_cq_inc_local, cqd->epoch_ptr);
        m_verb("complete_send: I have done %d write to %d, value sent = %llu",
               mem->ofi.sync.icntr[rank[i]], i, cqd->sync.data);
        // we use the user's MR but it doesn't matter as we provide a 0 length buffer
        struct iovec iov = {
            .iov_base = &cqd->sync.data,
            .iov_len = 0,
        };
        struct fi_rma_iov riov = {
            .addr = mem->ofi.mr.base_list[rank[i]],
            .len = 0,
            .key = mem->ofi.mr.key_list[rank[i]],
        };
        struct fi_msg_rma msg = {
            .msg_iov = &iov,
            .desc = &cqd->sync.mr.desc,
            .iov_count = 1,
            .addr = trx->addr[rank[i]],
            .context = &cqd->ctx,
            .rma_iov = &riov,
            .rma_iov_count = 1,
            .data = m_ofi_data_set_cmpl,  //| m_ofi_data_set_nops(issued_rank),
        };
        const uint64_t flag = FI_TRANSMIT_COMPLETE | FI_REMOTE_CQ_DATA;
        m_verb("complete using fi_writemsg with EP %p", trx->ep);
        m_ofi_call_again(fi_writemsg(trx->ep, &msg, flag), &progress);
    }
    return m_success;
}
//==================================================================================================
// ATOMICS
//==================================================================================================
int ofi_rmem_post_fiatomic(const int nrank, const int* rank, ofi_rmem_t* mem,
                                  ofi_comm_t* comm) {
    // notify readiness to the rank list, posting uses cqdata_ps to complete
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    mem->ofi.sync.ps_sig.inc = 1;
    for (int i = 0; i < nrank; ++i) {
        // used for completion
        ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps + i;
        cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
        // issue the atomic, local var
        struct fi_ioc iov = {
            .count = 1,  // depends on datatype
            .addr = &mem->ofi.sync.ps_sig.inc,
        };
        struct fi_rma_ioc rma_iov = {
            .count = 1,  // depends on datatype
            .addr = mem->ofi.sync.ps_sig.val_mr.base_list[rank[i]] + 0,
            .key = mem->ofi.sync.ps_sig.val_mr.key_list[rank[i]],
        };
        struct fi_msg_atomic msg = {
            .msg_iov = &iov,
            .desc = &mem->ofi.sync.ps_sig.inc_mr.desc,
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
        m_verb("atomic: inc = %d", mem->ofi.sync.ps_sig.inc);
        m_verb("cqdata ctx %p, kind = local %d, epoch_ptr = %p",&cqdata->ctx,cqdata->kind & m_ofi_cq_inc_local,cqdata->epoch_ptr);
        m_ofi_call_again(fi_atomicmsg(mem->ofi.sync_trx->ep, &msg, FI_TRANSMIT_COMPLETE),
                         &progress);
    }
    return m_success;
}
int ofi_rmem_start_fiatomic(const int nrank, const int* rank, ofi_rmem_t* mem,
                                   ofi_comm_t* comm) {
    // notify readiness to the rank list
    ofi_progress_t progress = {
        .cq = mem->ofi.sync_trx->cq,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
    };
    // used for completion, only the first one
    ofi_cqdata_t* cqdata = mem->ofi.sync.cqdata_ps;
    cqdata->kind = m_ofi_cq_kind_null | m_ofi_cq_inc_local;
    // issue the atomic
    int myself = comm->rank;
    // even we read it, we need to provide a source buffer
    struct fi_ioc iov = {
        .addr = &mem->ofi.sync.ps_sig.inc,
        .count = 1,
    };
    struct fi_ioc res_iov = {
        .addr = &mem->ofi.sync.ps_sig.res,
        .count = 1,
    };
    struct fi_rma_ioc rma_iov = {
        .count = 1,
        .addr = mem->ofi.sync.ps_sig.val_mr.base_list[myself] + 0,
        .key = mem->ofi.sync.ps_sig.val_mr.key_list[myself],
    };
    struct fi_msg_atomic msg = {
        .msg_iov = &iov,
        .desc = &mem->ofi.sync.ps_sig.inc_mr.desc,
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
    m_verb("cqdata ctx %p, kind = local %d, epoch_ptr = %p",&cqdata->ctx,cqdata->kind & m_ofi_cq_inc_local,cqdata->epoch_ptr);
    m_ofi_call_again(fi_fetch_atomicmsg(mem->ofi.sync_trx->ep, &msg, &res_iov,
                                        &mem->ofi.sync.ps_sig.res_mr.desc, 1, FI_TRANSMIT_COMPLETE),
                     &progress);
    return m_success;
}
//==================================================================================================
// WAIT
//==================================================================================================
/** @brief progress the cq in trx until the value of cntr has reached the threshold value
 * the epoch_ptr is used to handle special context when doing progress
 */
int ofi_rmem_progress_wait_noyield(const int threshold, countr_t* cntr, int n_trx, ofi_rma_trx_t* trx,
                           countr_t* epoch_ptr) {
    m_assert(n_trx > 0, "nothing to wait on");
    m_assert(trx, "nothing to wait on");
    //----------------------------------------------------------------------------------------------
    ofi_progress_t progress = {
        .cq = NULL,
        .xctx.epoch_ptr = epoch_ptr,
    };
    // loop on the array of trx and progress them untill the local completion value is reached
    int i = 0;
    while (m_countr_load(cntr) < threshold) {
        progress.cq = trx[i].cq;
        m_ofi_call(ofi_progress(&progress));
        i = (i + 1) % (n_trx);
    }
    // remove the values to the counter
    if (threshold) {
        m_countr_fetch_add(cntr, -threshold);
    }
    //----------------------------------------------------------------------------------------------
    return m_success;
}

//==================================================================================================
// ISSUE DTC tool functions
//==================================================================================================
int ofi_rmem_issue_dtc(rmem_complete_ack_t* ack) {
    // acknowledgement is submitted to the
    // send the ack if not delivery complete, if a fence is needed, it's added to the flag
    // directly in the call to (t)send
    switch (ack->comm->prov_mode.dtc_mode) {
        case (M_OFI_DTC_NULL):
            m_assert(0, "should not be NULL here");
            break;
        case (M_OFI_DTC_TAGGED):
            m_rmem_call(ofi_rmem_complete_fitsend(ack->nrank, ack->rank, ack->mem, ack->comm));
            break;
        case (M_OFI_DTC_MSG):
            m_rmem_call(ofi_rmem_complete_fisend(ack->nrank, ack->rank, ack->mem, ack->comm));
            break;
        case (M_OFI_DTC_CQDATA):
            m_rmem_call(ofi_rmem_complete_fiwrite(ack->nrank, ack->rank, ack->mem, ack->comm));
            break;
    };
    return m_success;
}

// end-of-file
