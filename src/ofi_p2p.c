/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "ofi.h"
#include <pmi.h>
#include "rmem_utils.h"
#include "ofi_utils.h"

static int ofi_p2p_init(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm, const p2p_opt_t op) {
    m_assert(ctx_id < comm->n_ctx, "ctx id = %d < the number of ctx = %d", ctx_id, comm->n_ctx);
    //----------------------------------------------------------------------------------------------
    ofi_ctx_t* ctx = comm->ctx + ctx_id;
    p2p->ofi.cq.kind = m_ofi_cq_kind_rqst;
    // init the data stuctures
    p2p->ofi.iov = (struct iovec){
        .iov_base = p2p->buf,
        .iov_len = p2p->count,
    };
    p2p->ofi.msg = (struct fi_msg_tagged){
        .msg_iov = &p2p->ofi.iov,
        .desc = &p2p->ofi.desc_local,
        .iov_count = 1,
        .addr = ctx->p2p_addr[p2p->peer],
        .tag = ofi_set_tag(ctx_id, p2p->tag),
        .ignore = 0x0,
        .context = &p2p->ofi.cq.ctx,
        .data = 0,
    };
    // store the ep
    p2p->ofi.kind = op;
    if (op == P2P_OPT_SEND) {
        p2p->ofi.ep = ctx->p2p_ep;
    } else {
        p2p->ofi.ep = ctx->srx;
    }
    // set the progress param
    p2p->ofi.progress.cq = ctx->p2p_cq;
    p2p->ofi.progress.fallback_ctx = NULL;
    // flag
    // we can use the inject (or FI_INJECT_COMPLETE) only if autoprogress cap is ON. Otherwise, not
    // reading the cq will lead to no progress, see issue https://github.com/pmodels/rmem/issues/4
    const bool auto_progress = (comm->prov->domain_attr->data_progress & FI_PROGRESS_AUTO);
    const bool do_inject = (p2p->count <= comm->prov->tx_attr->inject_size) && auto_progress;
    p2p->ofi.flags = FI_COMPLETION;
    p2p->ofi.flags |= do_inject ? FI_INJECT : 0x0;
    p2p->ofi.flags |= (auto_progress ? FI_INJECT_COMPLETE : FI_TRANSMIT_COMPLETE);
    //----------------------------------------------------------------------------------------------
    m_rmem_call(ofi_util_mr_reg(p2p->buf, p2p->count, FI_SEND | FI_RECV, comm, &p2p->ofi.mr_local,
                                &p2p->ofi.desc_local, NULL));
    m_rmem_call(ofi_util_mr_bind(ctx->p2p_ep, p2p->ofi.mr_local, NULL, comm));
    m_rmem_call(ofi_util_mr_enable(p2p->ofi.mr_local, comm, NULL));

    return m_success;
}

int ofi_send_init(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm) {
    m_rmem_call(ofi_p2p_init(p2p, ctx_id, comm, P2P_OPT_SEND));
    return m_success;
}

int ofi_recv_init(ofi_p2p_t* p2p, const int ctx_id, ofi_comm_t* comm) {
    m_rmem_call(ofi_p2p_init(p2p, ctx_id, comm, P2P_OPT_RECV));
    return m_success;
}

int ofi_p2p_start(ofi_p2p_t* p2p) {
    // busy counter
    m_countr_store(&p2p->ofi.cq.rqst.busy, 1);

    switch (p2p->ofi.kind) {
        case (P2P_OPT_SEND): {
            m_ofi_call_again(fi_tsendmsg(p2p->ofi.ep, &p2p->ofi.msg, p2p->ofi.flags),
                             &p2p->ofi.progress);
            if (p2p->ofi.flags & FI_INJECT) {
                m_countr_fetch_add(&p2p->ofi.cq.rqst.busy, -1);
            }
        } break;
        case (P2P_OPT_RECV): {
            uint64_t flag = FI_COMPLETION;
            m_ofi_call_again(fi_trecvmsg(p2p->ofi.ep, &p2p->ofi.msg, flag), &p2p->ofi.progress);
        } break;
    }
    return m_success;
}


int ofi_p2p_free(ofi_p2p_t* p2p) {
    m_rmem_call(ofi_util_mr_close(p2p->ofi.mr_local));
    return m_success;
}
