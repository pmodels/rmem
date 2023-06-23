/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "ofi.h"
#include "rmem_utils.h"
#include <inttypes.h>
#include <stdint.h>

#define m_ofi_cq_entries 16
#define m_ofi_cq_err_len 512

#define m_ofi_cq_offset(a) (offsetof(ofi_cqdata_t, a) - offsetof(ofi_cqdata_t, ctx))

static void ofi_cq_update_data(uint64_t* data, countr_t* epoch) {
    m_assert(*data > 0, "the value of data should be > 0 (and not %" PRIu64 ")", *data);
    m_assert(sizeof(int) == sizeof(uint32_t), "atomic int must be of size 32");
    uint32_t post = m_ofi_data_get_post(*data);
    uint32_t cmpl = m_ofi_data_get_cmpl(*data);
    uint32_t nops = m_ofi_data_get_nops(*data);

    // get the atomic_int array to increment, always the same one
    if (post > 0) {
        m_assert(post <= 1, "post must be <=1");
        m_countr_fetch_add(epoch + 0, post);
    }
    if (cmpl > 0) {
        m_assert(cmpl <= 1, "post must be <=1");
        m_countr_fetch_add(epoch + 2, nops);
        m_countr_fetch_add(epoch + 1, cmpl);
    }
}

int ofi_progress(ofi_cqdata_t* cq) {
    ofi_cq_entry event[m_ofi_cq_entries];
    int ret = fi_cq_read(cq->cq, event, m_ofi_cq_entries);
    if (ret > 0) {
        //------------------------------------------------------------------------------------------
        // entries in the buffer
        for (int i = 0; i < ret; ++i) {
            // get the context
            uint8_t* op_ctx = (uint8_t*)event[i].op_context;
            // if the context is null, the cq is used for remote data
            // uint64_t data = event[i].data;
            // if (!op_ctx) {
            //     m_verb("completion data received");
            //     ofi_cq_update_data(&data, cq->sync.cntr);
            // } else {
            m_assert(op_ctx, "the context cannot be null here");
            // recover the kind
            uint8_t kind = *((uint8_t*)op_ctx + m_ofi_cq_offset(kind));
            if (kind & m_ofi_cq_kind_rqst) {
                atomic_int** flag = (atomic_int**)(op_ctx + m_ofi_cq_offset(rqst.flag));
                if (*flag) {
                    atomic_fetch_add(*flag, 1);
                    m_verb("increasing flag value by 1");
                }
            } else if (kind & m_ofi_cq_kind_sync) {
                countr_t** epoch = (countr_t**)(op_ctx + m_ofi_cq_offset(sync.cntr));
                uint64_t* data = (uint64_t*)(op_ctx + m_ofi_cq_offset(sync.data));
                ofi_cq_update_data(data, *epoch);
            } else {
                m_assert(0, "unknown kind: %d", kind);
            }
            // }
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
        ssize_t ret_cqerr = fi_cq_readerr(cq->cq, &err, 0);
        if (ret_cqerr == -FI_EAGAIN) {
            m_log("no error to be read");
        } else {
            switch (err.err) {
                case (FI_ETRUNC):
                    m_log("truncated message");
                    break;
                default: {
                    char prov_err[m_ofi_cq_err_len];
                    fi_cq_strerror(cq->cq, err.prov_errno, err_data, prov_err, m_ofi_cq_err_len);
                    m_log("OFI-CQ ERROR: %s, provider says %s", fi_strerror(err.err), prov_err);
                }
            }
        }
        return m_failure;
    }
    return m_success;
}

int ofi_wait(ofi_cqdata_t* cq) {
    m_assert(cq->kind == m_ofi_cq_kind_rqst, "wrong kind of request");
    // while the request is not completed, progress the CQ
    while (!m_countr_load(cq->rqst.flag)) {
        m_rmem_call(ofi_progress(cq));
    }
    return m_success;
}
int ofi_p2p_wait(ofi_p2p_t* p2p) {
    m_rmem_call(ofi_wait(&p2p->ofi.cq));
    return m_success;
}
int ofi_rma_wait(ofi_rma_t* rma) {
    m_rmem_call(ofi_wait(&rma->ofi.cq));
    return m_success;
}

// end of file
