#include "ofi.h"
#include "ofi_utils.h"

int ofi_rmem_sig_wait(const uint32_t val, ofi_rmem_t* mem, ofi_comm_t* comm) {
    int i = 0;
    ofi_progress_t progress = {
        .cq = NULL,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
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
            // the EP here doesn't matter because the MR has been attached to all of them
            // so we randomly choose index 0 to check this
            progress.cq = mem->ofi.data_trx[0].cq;
            m_rmem_call(ofi_util_sig_wait(&mem->ofi.signal, comm->rank,
                                               mem->ofi.data_trx[0].addr[comm->rank],
                                               mem->ofi.data_trx[0].ep, &progress, val));
        } break;
    }
    return m_success;
}
