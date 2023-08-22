#include "ofi.h"

int ofi_rmem_sig_wait(const uint32_t val, ofi_rmem_t* mem, ofi_comm_t* comm) {
    int i = 0;
    ofi_progress_t progress = {
        .cq = NULL,
        // any cqdata will work, they all refer to the same epoch array
        .fallback_ctx = &mem->ofi.sync.cqdata_cw->ctx,
    };
    switch (comm->prov_mode.sig_mode) {
        case M_OFI_SIG_NULL:
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_SIG_CQ_DATA): {
            do {
                progress.cq = mem->ofi.data_trx[i].cq;
                m_rmem_call(ofi_progress(&progress));
                // go to the next cq
                i = (i + 1) % mem->ofi.n_rx;
            } while (m_countr_load(m_rma_mepoch_signal(mem)) < val);
            m_countr_fetch_add(m_rma_mepoch_signal(mem), -val);
        } break;
        case (M_OFI_SIG_ATOMIC): {
            m_verb("waiting for signal to be %d", val);
            m_ofi_call(fi_cntr_wait(mem->ofi.signal.scntr, val, -1));
            m_ofi_call(fi_cntr_set(mem->ofi.signal.scntr, 0));
        } break;
    }
    return m_success;
}
