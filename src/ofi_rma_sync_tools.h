#ifndef OFI_RMA_SYNC_TOOLS_
#define OFI_RMA_SYNC_TOOLS_
#include "ofi.h"
#include "rmem_qlist.h"

#ifndef NDEBUG
#define m_mem_check_empty_cq(cq)                                                    \
    do {                                                                            \
        ofi_cq_entry event[1];                                                      \
        int ret = fi_cq_read(cq, event, 1);                                         \
        uint8_t* op_ctx = (uint8_t*)event[0].op_context;                            \
        uint8_t kind;                                                               \
        if (ret > 0) {                                                              \
            kind = *((uint8_t*)op_ctx +                                             \
                     (offsetof(ofi_cqdata_t, kind) - offsetof(ofi_cqdata_t, ctx))); \
        }                                                                           \
        m_assert(ret <= 0, "ret = %d, cq is NOT empty, kind = %u", ret, kind);      \
    } while (0)
#else
#define m_mem_check_empty_cq(cq) \
    { ((void)0); }

#endif

//==================================================================================================
typedef struct {
    rmem_lnode_t node;
    const int nrank;
    const int* rank;
    ofi_rmem_t* mem;
    ofi_comm_t* comm;
} rmem_complete_ack_t;
int ofi_rmem_issue_dtc(rmem_complete_ack_t* ack);

//==================================================================================================
int ofi_rmem_am_init(ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_am_free(ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_am_repost(ofi_cqdata_t* cqdata, ofi_progress_t* progress);
//==================================================================================================
// int ofi_rmem_start_firecv(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_start_fitrecv(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_start_fiatomic(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);

//==================================================================================================
int ofi_rmem_post_fitsend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_post_fisend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_post_fiatomic(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);

//==================================================================================================
int ofi_rmem_complete_fitsend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_complete_fisend(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_complete_fiwrite(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
int ofi_rmem_wait_fitrecv(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm);
//==================================================================================================
int ofi_rmem_progress_wait_noyield(const int threshold, countr_t* cntr, int n_trx, ofi_rma_trx_t* trx,
                           countr_t* epoch_ptr);
#endif
