/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "ofi.h"
#include <inttypes.h>
#include <stdint.h>

#define m_ofi_cq_entries 16
#define m_ofi_cq_err_len 512

#define m_ofi_cq_offset(a) (offsetof(ofi_cqdata_t, a) - offsetof(ofi_cqdata_t, ctx))

static void ofi_cq_update_sync_tag(uint64_t* data, countr_t* epoch) {
    m_assert(*data > 0, "the value of data should be > 0 (and not %" PRIu64 ")", *data);
    m_assert(sizeof(int) == sizeof(uint32_t), "atomic int must be of size 32");
    // get the atomic_int array to increment, always the same one
    uint32_t post = m_ofi_data_get_post(*data);
    if (post > 0) {
        m_assert(post <= 1, "post must be <=1");
        m_countr_fetch_add(m_rma_epoch_post(epoch), post);
        // if we get a post, we do not need to handle something else
        return;
    }
    uint32_t cmpl = m_ofi_data_get_cmpl(*data);
    uint32_t nops = m_ofi_data_get_nops(*data);
    if (cmpl > 0) {
        m_assert(cmpl <= 1, "post must be <=1");
        m_countr_fetch_add(m_rma_epoch_remote(epoch), nops);
        m_countr_fetch_add(m_rma_epoch_cmpl(epoch), cmpl);
        m_verb("sync: counter +%d, now = %d",nops,m_countr_load(m_rma_epoch_remote(epoch)));
        // if we get a complete, we do not need to handle something else
        return;
    }
    uint32_t rcqd = m_ofi_data_get_rcq(*data);
    if (rcqd > 0) {
        // if we receive a remote cq data then we remove -1 to the epoch[2]
        m_assert(rcqd <= 1, "post must be <=1");
        m_countr_fetch_add(m_rma_epoch_remote(epoch), -1);
        m_verb("remote data: counter -1, now = %d",m_countr_load(m_rma_epoch_remote(epoch)));
    }
#if (M_WRITE_DATA)
    uint32_t sig = m_ofi_data_get_sig(*data);
    if (sig > 0) {
        m_assert(sig <= 1, "post must be <=1");
        m_countr_fetch_add(m_rma_epoch_signal(epoch), 1);
    }
#endif
    return;
}

/**
 * @brief progress the endpoint associated to the CQ and read the entries if any 
 *
 * if the context provided is NULL and !M_SYNC_RMA_EVENT, use the null_ctx pointer
*/
int ofi_progress(ofi_progress_t* progress) {
    struct fid_cq* cq = progress->cq;
    ofi_cq_entry event[m_ofi_cq_entries];
    int ret = fi_cq_read(cq, event, m_ofi_cq_entries);
    if (ret > 0) {
        //------------------------------------------------------------------------------------------
        // entries in the buffer
        for (int i = 0; i < ret; ++i) {
            m_verb("processing #%d/%d",i,ret);
            // get the context
            uint8_t* op_ctx = (uint8_t*)event[i].op_context;
#if (!M_SYNC_RMA_EVENT || M_WRITE_DATA)
            // is it a remote data?
            if (!op_ctx) {
                m_verb("data entry completed: using the fallback");
                op_ctx = progress->fallback_ctx;
                countr_t** epoch = (countr_t**)(op_ctx + m_ofi_cq_offset(sync.cntr));
                uint64_t data = event[i].data;
                ofi_cq_update_sync_tag(&data, *epoch);
                continue;
            }
#endif
            // if the context is null, the cq is used for remote data
            m_assert(op_ctx, "the context cannot be null here");
            // recover the kind
            uint8_t kind = *((uint8_t*)op_ctx + m_ofi_cq_offset(kind));
            if (kind & m_ofi_cq_inc_local) {
                countr_t** epoch = (countr_t**)(progress->fallback_ctx+ m_ofi_cq_offset(sync.cntr));
                m_assert(*epoch,"epoch is null, that's annoying");
                m_countr_fetch_add(m_rma_epoch_local(*epoch), 1);
            }
            if (kind & m_ofi_cq_kind_rqst) {
                m_verb("rqst entry completed");
                countr_t* cntr = (countr_t*)(op_ctx + m_ofi_cq_offset(rqst.busy));
                m_countr_fetch_add(cntr,-1);
                m_verb("decreasing flag value by -1");
                continue;
            } else if (kind & m_ofi_cq_kind_sync) {
                m_verb("sync entry completed");
                countr_t** epoch = (countr_t**)(op_ctx + m_ofi_cq_offset(sync.cntr));
                uint64_t* data = (uint64_t*)(op_ctx + m_ofi_cq_offset(sync.buf));
                ofi_cq_update_sync_tag(data, *epoch);
                continue;
            } else if (kind & m_ofi_cq_kind_null) {
                m_verb("null request completed");
                continue;
            } else {
                m_assert(0, "unknown kind: %d", kind);
            }
        }
    } else if (ret == (-FI_EAGAIN)) {
        //------------------------------------------------------------------------------------------
        // no entry
        return m_success;
    } else {
        //------------------------------------------------------------------------------------------
        // errors
        char err_data[m_ofi_cq_err_len];
        struct fi_cq_err_entry err = {
            .err_data = err_data,
            .err_data_size = sizeof(err_data),
        };
        ssize_t ret_cqerr = fi_cq_readerr(cq, &err, 0);
        if (ret_cqerr == -FI_EAGAIN) {
            m_log("no error to be read");
        } else {
            switch (err.err) {
                case (FI_ETRUNC):
                    m_log("truncated message");
                    break;
                default: {
                    char prov_err[m_ofi_cq_err_len];
                    fi_cq_strerror(cq, err.prov_errno, err_data, prov_err, m_ofi_cq_err_len);
                    m_log("OFI-CQ ERROR: %s, provider says %s", fi_strerror(err.err), prov_err);
                }
            }
        }
        return m_failure;
    }
    return m_success;
}

int ofi_wait(countr_t* busy, ofi_progress_t* progress) {
    // while the request is not completed, progress the CQ
    while (m_countr_load(busy)) {
        m_rmem_call(ofi_progress(progress));
    }
    return m_success;
}
int ofi_p2p_wait(ofi_p2p_t* p2p) {
    m_assert(p2p->ofi.cq.kind == m_ofi_cq_kind_rqst, "wrong kind of request");
    m_rmem_call(ofi_wait(&p2p->ofi.cq.rqst.busy, &p2p->ofi.progress));
    return m_success;
}
int ofi_rma_wait(ofi_rma_t* rma) {
    m_assert(rma->ofi.msg.cq.kind == m_ofi_cq_kind_rqst, "wrong kind of request");
    m_rmem_call(ofi_wait(&rma->ofi.msg.cq.rqst.busy, &rma->ofi.progress));
    return m_success;
}

// end of file
