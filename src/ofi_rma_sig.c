#include "ofi.h"

int ofi_rmem_sig_wait(const uint32_t val, ofi_rmem_t* mem) {
    int i = 0;
    ofi_progress_t progress = {
        .cq = NULL,
        // any cqdata will work, they all refer to the same epoch array
        .fallback_ctx = &mem->ofi.sync.cqdata->ctx,
    };
#if (M_WRITE_DATA)
    while (m_countr_load(mem->ofi.sync.epoch + 3) < val) {
        progress.cq = mem->ofi.data_trx[i].cq;
        m_rmem_call(ofi_progress(&progress));
        // go to the next cq
        i = (i + 1) % mem->ofi.n_rx;
    }
    m_countr_store(mem->ofi.sync.epoch + 3, 0);
#else
    m_verb("waiting for signal to be %d",val);
    m_ofi_call(fi_cntr_wait(mem->ofi.signal.scntr,val,-1));
    m_ofi_call(fi_cntr_set(mem->ofi.signal.scntr,0));
#endif
    return m_success;
}
