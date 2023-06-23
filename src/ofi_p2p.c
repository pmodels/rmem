/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "ofi.h"
#include "rmem_utils.h"

int ofi_p2p_create(ofi_p2p_t* p2p, ofi_comm_t* ofi) {
    p2p->ofi.cq.kind = m_ofi_cq_kind_rqst;
    // register the flag
    p2p->ofi.cq.rqst.flag = &p2p->ofi.completed;
    // init the data stuctures
    p2p->ofi.iov = (struct iovec){
        .iov_base = p2p->buf,
        .iov_len = p2p->count,
    };
    p2p->ofi.msg = (struct fi_msg_tagged){
        .msg_iov = &p2p->ofi.iov,
        .desc = NULL,
        .iov_count = 1,
        .ignore = 0x0,
        .context = &p2p->ofi.cq.ctx,
        .data = 0,
    };
    return m_success;
}

typedef enum {
    P2P_OPT_SEND,
    P2P_OPT_RECV,
} p2p_opt_t;

int ofi_p2p_enqueue(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm, const p2p_opt_t op) {
    m_assert(ctx_id < comm->n_ctx, "ctx id = %d < the number of ctx = %d", ctx_id, comm->n_ctx);
    ofi_ctx_t* ctx = comm->ctx + ctx_id;

    // set the completion param
    p2p->ofi.cq.cq = ctx->p2p_cq;
    m_countr_store(p2p->ofi.cq.rqst.flag, 0);
    // address and tag depends on the communicator context
    p2p->ofi.msg.tag = ofi_set_tag(ctx_id, p2p->tag);
    p2p->ofi.msg.addr = ctx->p2p_addr[p2p->peer];

    // we can use the inject (or FI_INJECT_COMPLETE) only if autoprogress cap is ON. Otherwise, not
    // reading the cq will lead to no progress, see issue https://github.com/pmodels/rmem/issues/4
    const bool auto_progress = (comm->prov->domain_attr->data_progress & FI_PROGRESS_AUTO);
    const bool do_inject = (p2p->count <= comm->prov->tx_attr->inject_size) && auto_progress;
    switch (op) {
        case (P2P_OPT_SEND): {
            uint64_t flag =
                (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE) | FI_COMPLETION;
            if (do_inject) {
                m_ofi_call(fi_tinject(ctx->p2p_ep, p2p->buf, p2p->count, ctx->p2p_addr[p2p->peer],
                                      p2p->ofi.msg.tag));
                // need to complete the request as no CQ entry will happen
                m_countr_fetch_add(p2p->ofi.cq.rqst.flag, 1);
            } else {
                m_ofi_call(fi_tsendmsg(ctx->p2p_ep, &p2p->ofi.msg, flag));
                // m_ofi_call(fi_tsend(ctx->p2p_ep, p2p->buf, p2p->count, NULL,
                //                     ctx->p2p_addr[p2p->peer], tag, &p2p->ofi.cq.ctx));
            }
        } break;
        case (P2P_OPT_RECV): {
            uint64_t flag = FI_COMPLETION;
            m_ofi_call(fi_trecvmsg(ctx->srx, &p2p->ofi.msg, flag));
        } break;
    }
    return m_success;
}

int ofi_send_enqueue(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm) {
    m_rmem_call(ofi_p2p_enqueue(p2p, ctx_id, comm, P2P_OPT_SEND));
    return m_success;
}

int ofi_recv_enqueue(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm) {
    m_rmem_call(ofi_p2p_enqueue(p2p, ctx_id, comm, P2P_OPT_RECV));
    return m_success;
}

int ofi_p2p_free(ofi_p2p_t* p2p) { return m_success; }
