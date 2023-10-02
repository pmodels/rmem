/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef OFI_UTILS_H_
#define OFI_UTILS_H_

#include <rdma/fabric.h>
#include <stdbool.h>
#include "ofi.h"

// capabilities are defined over 8 bits
typedef uint8_t ofi_cap_t;
#define M_OFI_PROV_HAS_FENCE     0x01  // 0000 0001
#define M_OFI_PROV_HAS_ATOMIC    0x02  // 0000 0010
#define M_OFI_PROV_HAS_CQ_DATA   0x04  // 0000 0100
#define M_OFI_PROV_HAS_RMA_EVENT 0x08  // 0000 1000

#define m_ofi_prov_has_fence(a)     ((a)&M_OFI_PROV_HAS_FENCE)
#define m_ofi_prov_has_atomic(a)    ((a)&M_OFI_PROV_HAS_ATOMIC)
#define m_ofi_prov_has_cq_data(a)   ((a)&M_OFI_PROV_HAS_CQ_DATA)
#define m_ofi_prov_has_rma_event(a) ((a)&M_OFI_PROV_HAS_RMA_EVENT)


int ofi_util_get_prov(struct fi_info** prov, ofi_mode_t* prov_mode);
int ofi_util_new_ep(const bool new_ctx, struct fi_info* prov, struct fid_domain* dom,
                    struct fid_ep** ep, struct fid_stx** stx, struct fid_ep** srx);
int ofi_util_free_ep(struct fi_info* prov, struct fid_ep** ep, struct fid_stx** stx,
                     struct fid_ep** srx);
int ofi_util_av(const int n_addr, struct fid_ep* ep, struct fid_av* av, fi_addr_t** addr);

int ofi_util_mr_reg(void* buf, size_t count, uint64_t access, ofi_comm_t* comm, struct fid_mr** mr,
                    void** desc, uint64_t** base_list);
int ofi_util_mr_bind(struct fid_ep* ep, struct fid_mr* mr, struct fid_cntr* cntr, ofi_comm_t* comm);
int ofi_util_mr_enable(struct fid_mr* mr, ofi_comm_t* comm, uint64_t** key_list);
int ofi_util_mr_close(struct fid_mr* mr);

int ofi_util_sig_reg(ofi_mem_sig_t* sig, ofi_comm_t* comm);
int ofi_util_sig_bind(ofi_mem_sig_t* sig, struct fid_ep* ep, ofi_comm_t* comm);
int ofi_util_sig_enable(ofi_mem_sig_t* sig, ofi_comm_t* comm);
int ofi_util_sig_close(ofi_mem_sig_t* sig);
int ofi_util_sig_wait(ofi_mem_sig_t* sig, int myrank, fi_addr_t myaddr, struct fid_ep* ep,
                      ofi_progress_t* progress, uint32_t threshold);
#endif
