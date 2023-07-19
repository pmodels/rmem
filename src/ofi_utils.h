/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef OFI_UTILS_H_
#define OFI_UTILS_H_

#include <rdma/fabric.h>
#include <stdbool.h>
#include "ofi.h"

int ofi_util_get_prov(struct fi_info** prov);
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

#endif
