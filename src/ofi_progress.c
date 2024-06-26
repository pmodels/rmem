/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
#include "ofi.h"
#include "ofi_rma_sync_tools.h"
#include "rmem_utils.h"
#include <inttypes.h>
#include <stdint.h>

#define m_ofi_cq_err_len 512

#define m_ofi_cq_offset(a) (offsetof(ofi_cqdata_t, a) - offsetof(ofi_cqdata_t, ctx))

static void ofi_cq_update_sync_tag(uint64_t* data, countr_t* epoch) {
    m_assert(data, "the data must point to something: %p", data);
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
        m_assert(cmpl <= 1, "cmpl must be <=1");
        m_countr_fetch_add(m_rma_epoch_cmpl(epoch), cmpl);
        if (nops > 0) {
            m_countr_fetch_add(m_rma_epoch_remote(epoch), -nops);
        }
        m_verb("sync: counter -%d, now = %d",nops,m_countr_load(m_rma_epoch_remote(epoch)));
        // if we get a complete, we do not need to handle something else
        return;
    }
    uint32_t rcqd = m_ofi_data_get_rcq(*data);
    if (rcqd > 0) {
        // if we receive a remote cq data then we add +1 to the epoch[2]
        m_assert(rcqd <= 1, "post must be <=1");
        m_countr_fetch_add(m_rma_epoch_remote(epoch), +1);
        m_verb("remote cq data: counter +1, now = %d",m_countr_load(m_rma_epoch_remote(epoch)));
    }
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
            m_verb("processing #%d/%d", i, ret);
            // first we have to treat the FI_MULTI_RECV case
            // get the context
            uint8_t* op_ctx = (uint8_t*)event[i].op_context;
            //--------------------------------------------------------------------------------------
            // is it a remote data?
            if (!op_ctx) {
                countr_t* epoch_ptr = progress->xctx.epoch_ptr;
                uint64_t data = event[i].data;
                m_verb("data entry completed: using the fallback with data = %llu", data);
                m_assert(epoch_ptr, "epoch is null, that's annoying");
                ofi_cq_update_sync_tag(&data, epoch_ptr);
            } else {
                //----------------------------------------------------------------------------------
                // recover the kind
                m_assert(op_ctx, "the context cannot be null here");
                uint8_t kind = *((uint8_t*)op_ctx + m_ofi_cq_offset(kind));
                //----------------------------------------------------------------------------------
                // local can be layer on top of all the other kinds
                if (likely(kind & m_ofi_cq_inc_local)) {
                    countr_t** epoch = (countr_t**)(op_ctx + m_ofi_cq_offset(epoch_ptr));
                    m_verb("local completion: ctx = %p, epoch = %p, kind = %d",op_ctx,*epoch,kind&0xf);
                    m_assert(*epoch, "epoch is null, that's annoying");
                    m_countr_fetch_add(m_rma_epoch_local(*epoch), 1);
                }
                //----------------------------------------------------------------------------------
                if (likely(kind & m_ofi_cq_kind_rqst)) {
                    m_verb("rqst entry completed");
                    countr_t* cntr = (countr_t*)(op_ctx + m_ofi_cq_offset(rqst.busy));
                    m_countr_fetch_add(cntr, -1);
                    m_verb("decreasing flag value by -1");
                    continue;
                } else if (likely(kind & m_ofi_cq_kind_sync)) {
                    m_verb("sync entry completed");
                    countr_t** epoch = (countr_t**)(op_ctx + m_ofi_cq_offset(epoch_ptr));
                    uint64_t* data = (uint64_t*)(op_ctx + m_ofi_cq_offset(sync.data));
                    ofi_cq_update_sync_tag(data, *epoch);
                    continue;
                } else if (likely(kind & m_ofi_cq_kind_am)) {
                    m_verb("am entry completed");
                    // we might have received a msg or the buffer is expired (or both)
                    if (event[i].flags & FI_RECV) {
                        m_verb("am recv completed");
                        m_assert(event[i].len == sizeof(uint64_t), "The size here should be 64");
                        countr_t** epoch = (countr_t**)(op_ctx + m_ofi_cq_offset(epoch_ptr));
                        uint64_t* data = (uint64_t*)(event[i].buf);
                        ofi_cq_update_sync_tag(data, *epoch);
                    }
                    if (event[i].flags & FI_MULTI_RECV) {
                        m_verb("buffer expired completed, reposting");
                        ofi_cqdata_t* cqdata =
                            (ofi_cqdata_t*)(op_ctx - offsetof(ofi_cqdata_t, ctx));
                        m_rmem_call(ofi_rmem_am_repost(cqdata, progress));
                    }
                } else if (unlikely(kind & m_ofi_cq_kind_null)) {
                    m_verb("null request completed");
                    continue;
                } else {
                    m_assert(0, "unknown kind: %d", kind);
                }
            }
        }
        return m_success;
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
                    m_log("OFI-CQ ERROR: truncated message %lu could not fit in AM buffers",
                          err.olen);
                    break;
                case (FI_ECANCELED):
                    m_log("OFI-CQ ERROR: operation has been successfully canceled: ctx = %p",
                          err.op_context);
                    break;
                default: {
                    char prov_err[m_ofi_cq_err_len];
                    fi_cq_strerror(cq, err.prov_errno, err_data, prov_err, m_ofi_cq_err_len);
                    m_log("OFI-CQ ERROR: %s, provider says: %s", fi_strerror(err.err), prov_err);
                }
            }
        }
        PrintBackTrace();
        PMI_Abort(EXIT_FAILURE, NULL);
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
