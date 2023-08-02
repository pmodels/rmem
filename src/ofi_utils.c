/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "ofi_utils.h"

#include <string.h>

#include "ofi.h"
#include "pmi_utils.h"
#include "rmem_utils.h"

#define ofi_str_len 128

#define m_ofi_test_info(hint, field, value)                               \
    do {                                                                  \
        uint64_t testvalue = hint->field & (value);                       \
        if ((testvalue) ^ (value)) {                                      \
            struct fi_info* test_prov;                                    \
            hint->field |= value;                                         \
            fi_getinfo(fi_version(), NULL, NULL, 0ULL, hint, &test_prov); \
            if (!test_prov) {                                             \
                m_log("imposible to set" #field " to " #value "> ooops"); \
                hints->field &= ~(value);                                 \
            } else {                                                      \
                fi_freeinfo(test_prov);                                   \
                m_verb("successfully set " #field " to " #value);         \
            }                                                             \
        } else {                                                          \
            m_verb(#field " already has " #value);                        \
        }                                                                 \
    } while (0)

#define m_ofi_fatal_info(hint, field, value)                              \
    do {                                                                  \
        uint64_t testvalue = hint->field & (value);                       \
        if ((testvalue) ^ (value)) {                                      \
            struct fi_info* test_prov;                                    \
            hint->field |= value;                                         \
            fi_getinfo(fi_version(), NULL, NULL, 0ULL, hint, &test_prov); \
            if (!test_prov) {                                             \
                m_assert(0, "imposible to set" #field " to " #value);     \
            } else {                                                      \
                fi_freeinfo(test_prov);                                   \
                m_verb("successfully set " #field " to " #value);          \
            }                                                             \
        } else {                                                          \
            m_verb(#field " already has " #value);                         \
        }                                                                 \
    } while (0)

typedef enum {
    M_OFI_EP_KIND_NORMAL,
    M_OFI_EP_KIND_SHARED,
} ofi_ep_kind_t;

int ofi_prov_score(char* provname) {
    if (0 == strcmp(provname, "cxi")) {
        return 3;
    } else if (0 == strcmp(provname, "psm3")) {
        return 2;
    } else if (0 == strcmp(provname, "sockets")) {
        return 1;
    }
    return 0;
}

#if (M_WRITE_DATA)
#define ofi_sig_cap 0x0
#else
#define ofi_sig_cap FI_FENCE | FI_ATOMIC
#endif
#if (M_SYNC_RMA_EVENT)
#define ofi_syn_cap FI_RMA_EVENT
#else
#define ofi_syn_cap 0x0
#endif
#define ofi_cap FI_MSG | FI_TAGGED | FI_RMA | FI_DIRECTED_RECV | ofi_sig_cap | ofi_syn_cap
// #define ofi_cap_ops_tx   FI_READ | FI_WRITE | FI_SEND | FI_ATOMIC
// #if (M_SYNC_RMA_EVENT)
// #define ofi_cap_ops_rx \
//     FI_ATOMIC | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RECV | FI_RMA_EVENT | FI_DIRECTED_RECV
// #else
// #define ofi_cap_ops_rx \
//     FI_ATOMIC | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RECV | FI_REMOTE_CQ_DATA | FI_DIRECTED_RECV
// #endif

int ofi_util_get_prov(struct fi_info** prov) {
    // get the list of available providers and select the best one
    int ofi_ver = fi_version();
    int best_score = -1;
    struct fi_info* best_prov = NULL;
    struct fi_info* prov_list = NULL;
    m_ofi_call(fi_getinfo(ofi_ver, NULL, NULL, 0ULL, NULL, &prov_list));
    for (struct fi_info* cprov = prov_list; cprov; cprov = cprov->next) {
        int score = ofi_prov_score(cprov->fabric_attr->prov_name);
        if (score > best_score) {
            best_score = score;
            best_prov = cprov;
        }
    }
    char* prov_name = best_prov->fabric_attr->prov_name;
    m_verb("best provider is %s", prov_name);

    // get the best provider's name
    struct fi_info* hints = fi_allocinfo();
    hints->fabric_attr->prov_name = malloc(strlen(prov_name) + 1);
    strcpy(hints->fabric_attr->prov_name, prov_name);
    fi_freeinfo(prov_list);  // no need of prov_list anymore
    
    // set the mode bits to 1, not doing this leads to provider selection failure
 //    hints->mode = ~0;
	// hints->domain_attr->mode = ~0;
	// hints->domain_attr->mr_mode = ~(FI_MR_BASIC | FI_MR_SCALABLE);

    //----------------------------------------------------------------------------------------------
    // basic requirement are the modes and the caps
    // get_info is free to waive the mode bit set, but they are supported
    hints->mode = 0x0;
    hints->domain_attr->mode = 0x0;
    hints->domain_attr->mr_mode = 0x0;
    // we provide the context
    hints->mode |= FI_CONTEXT;
    // do not bind under the same cq/counter different capabilities endpoints
    hints->mode |= FI_RESTRICTED_COMP;
    hints->domain_attr->mode |= FI_RESTRICTED_COMP; 
    // MR endpoint is supported
    hints->domain_attr->mr_mode |= FI_MR_ENDPOINT;
    // optional:
    //hints->domain_attr->mr_mode |= FI_MR_LOCAL;
    hints->domain_attr->mr_mode |= FI_MR_PROV_KEY;
    hints->domain_attr->mr_mode |= FI_MR_ALLOCATED;
    hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR;

    // enable GPU memory if needed and try to get those modes with the minimum caps
#if (M_HAVE_CUDA)
    hints->domain_attr->mr_mode |= FI_MR_HMEM;
    m_ofi_fatal_info(hints, caps, ofi_cap | FI_HMEM);
#else
    m_ofi_fatal_info(hints, caps, ofi_cap);  // implies SEND/RECV
#endif


    // improve the selection if required in case of specific usage
#if (M_SYNC_RMA_EVENT)
    // request support for MR_RMA_EVENT
    hints->domain_attr->mr_mode |= FI_MR_RMA_EVENT;
    m_ofi_fatal_info(hints, caps, FI_RMA_EVENT | FI_REMOTE_READ | FI_REMOTE_WRITE);
#endif
#if (!M_WRITE_DATA)
    m_ofi_fatal_info(hints, caps, FI_ATOMIC | FI_FENCE);  // implies (REMOTE_)READ/WRITE
#endif
    m_ofi_fatal_info(hints, caps, FI_ATOMIC);  // implies (REMOTE_)READ/WRITE

    //----------------------------------------------------------------------------------------------
    // try to get more specific behavior
    // Reliable Datagram
    m_ofi_test_info(hints, ep_attr->type, FI_EP_RDM);
    // try to use shared context (reduces the memory)
    m_ofi_test_info(hints, ep_attr->tx_ctx_cnt, FI_SHARED_CONTEXT);
    m_ofi_test_info(hints, ep_attr->rx_ctx_cnt, FI_SHARED_CONTEXT);
    // thread safe is the most expensive, yet secure one
    m_ofi_test_info(hints, domain_attr->threading, FI_THREAD_SAFE);
    // TODO: switch to thread_domain
    // m_ofi_test_info(hints, domain_attr->threading, FI_THREAD_DOMAIN);
    // enable automatic ressource management
    m_ofi_test_info(hints, domain_attr->resource_mgmt, FI_RM_ENABLED);
    m_ofi_test_info(hints, rx_attr->total_buffered_recv, 0);
    // m_ofi_test_info(hints, rx_attr->total_buffered_recv, 0);
    // request manual progress (comment when using sockets on MacOs)
    m_ofi_test_info(hints, domain_attr->data_progress, FI_PROGRESS_MANUAL);
    m_ofi_test_info(hints, domain_attr->control_progress, FI_PROGRESS_MANUAL);
    // no order required
    m_ofi_test_info(hints, tx_attr->msg_order, FI_ORDER_NONE);
    m_ofi_test_info(hints, rx_attr->msg_order, FI_ORDER_NONE);
    m_ofi_test_info(hints, tx_attr->comp_order, FI_ORDER_NONE);
    m_ofi_test_info(hints, rx_attr->comp_order, FI_ORDER_NONE);

    // check the mode arguments now, fail is some modes are required
    m_ofi_call(fi_getinfo(ofi_ver, NULL, NULL, 0ULL, hints, prov));
    m_assert(*prov, "The provider list is empty");
    m_assert(!((*prov)->mode & FI_RX_CQ_DATA), "need to use FI_RX_CQ_DATA");
    m_assert(!((*prov)->mode & FI_ASYNC_IOV), "need to use FI_ASYNC_IOV");
    m_assert(!((*prov)->domain_attr->mr_mode & FI_MR_RAW), "need to use FI_MR_RAW");

    // improsing the modes must happen on (*prov), otherwise it's overwritten when done in hints
    m_verb("%s: is FI_MR_LOCAL required? %d", (*prov)->fabric_attr->prov_name,
          ((*prov)->domain_attr->mr_mode & FI_MR_LOCAL) > 0);
    m_verb("%s: is FI_MR_PROV_KEY required? %d", (*prov)->fabric_attr->prov_name,
          ((*prov)->domain_attr->mr_mode & FI_MR_PROV_KEY) > 0);
    m_verb("%s: is FI_MR_ALLOCATED required? %d", (*prov)->fabric_attr->prov_name,
          ((*prov)->domain_attr->mr_mode & FI_MR_ALLOCATED) > 0);
    m_verb("%s: is FI_MR_VIRT_ADDR required? %d", (*prov)->fabric_attr->prov_name,
          ((*prov)->domain_attr->mr_mode & FI_MR_VIRT_ADDR) > 0);
    m_verb("found compatible provider: %s", (*prov)->fabric_attr->prov_name);

    // free the hints
    fi_freeinfo(hints);
    return m_success;
}

/**
 * @brief creates a new (shared or normal) endpoint associated to the domain
 *
 * If shared context is supported by the provider, it optionally create new contexts and associate
 * them to the endpoint.
 *
 * @param[in] new_ctx create new shared rx and tx contexts
 * @param[out] ep return the created endpoint
 * @param[in,out] stx the shared transmit context to associate to the endpoint if not NULL. Will be
 * overwritten if new_ctx is true.
 * @param[in,out] srx the shared receive context to associate to the endpoint if not NULL. To be
 * used to queue the receive requests. Will be overwritten if new_ctx is true.
 *
 *
 */
int ofi_util_new_ep(const bool new_ctx, struct fi_info* prov, struct fid_domain* dom,
                    struct fid_ep** ep, struct fid_stx** stx, struct fid_ep** srx) {
    // get the kind of endpoints to create
    ofi_ep_kind_t kind = (prov->ep_attr->rx_ctx_cnt == FI_SHARED_CONTEXT &&
                          prov->ep_attr->tx_ctx_cnt == FI_SHARED_CONTEXT)
                             ? M_OFI_EP_KIND_SHARED
                             : M_OFI_EP_KIND_NORMAL;

    // normal endpoints don't share any CTX
    if (kind == M_OFI_EP_KIND_NORMAL) {
        m_verb("creating normal EP");
        // m_assert(prov->tx_attr->msg_order == FI_ORDER_NONE, "no order for the msg order");
        // m_assert(prov->rx_attr->msg_order == FI_ORDER_NONE, "no order for the msg order");
        // m_assert(prov->tx_attr->comp_order == FI_ORDER_NONE, "no order for the comp order");
        // m_assert(prov->rx_attr->comp_order == FI_ORDER_NONE, "no order for the comp order");
        // only one rx context and tx_context per EP
        prov->ep_attr->rx_ctx_cnt = 1;
        prov->ep_attr->tx_ctx_cnt = 1;
        m_ofi_call(fi_endpoint(dom, prov, ep, NULL));
        // assign the EP to the srx to be transparent to the application
        *srx = *ep;
    } else if (kind == M_OFI_EP_KIND_SHARED) {
        m_verb("creating shared EP");
        // sanity checks
        m_assert(prov->ep_attr->rx_ctx_cnt & FI_SHARED_CONTEXT, "EP need shared ctx cap");
        m_assert(prov->ep_attr->tx_ctx_cnt & FI_SHARED_CONTEXT, "EP need shared ctx cap");
        // create shared Tx only if not provided
        if (new_ctx) {
            m_verb("creating new STX/SRX");
            // create the shared receive and transmit context
            struct fi_tx_attr tx_attr = {
                .caps = ofi_cap,
                .msg_order = FI_ORDER_NONE,
                .comp_order = FI_ORDER_NONE,
            };
            m_ofi_call(fi_stx_context(dom, &tx_attr, stx, NULL));
            struct fi_rx_attr rx_attr = {
                .caps = ofi_cap,
                .msg_order = FI_ORDER_NONE,
                .comp_order = FI_ORDER_NONE,
            };
            m_ofi_call(fi_srx_context(dom, &rx_attr, srx, NULL));
        }
        // point to point specific resources
        m_assert(prov->ep_attr->rx_ctx_cnt & FI_SHARED_CONTEXT, "endpoint need a shared cap");
        m_assert(prov->ep_attr->tx_ctx_cnt & FI_SHARED_CONTEXT, "endpoint need a shared cap");
        m_ofi_call(fi_endpoint(dom, prov, ep, NULL));

        // bind the srx context
        if (*srx) {
            m_verb("binding SRX");
            m_ofi_call(fi_ep_bind(*ep, &(*srx)->fid, 0));
        }
        if (*stx) {
            m_verb("binding STX");
            m_ofi_call(fi_ep_bind(*ep, &(*stx)->fid, 0));
        }
    }
    return m_success;
}

int ofi_util_free_ep(struct fi_info* prov, struct fid_ep** ep, struct fid_stx** stx,
                     struct fid_ep** srx) {
    // get the kind of endpoints to create
    ofi_ep_kind_t kind = (prov->ep_attr->rx_ctx_cnt == FI_SHARED_CONTEXT &&
                          prov->ep_attr->tx_ctx_cnt == FI_SHARED_CONTEXT)
                             ? M_OFI_EP_KIND_SHARED
                             : M_OFI_EP_KIND_NORMAL;
    if (kind == M_OFI_EP_KIND_SHARED && *stx) {
        m_ofi_call(fi_close(&(*stx)->fid));
    }
    if (kind == M_OFI_EP_KIND_SHARED && *srx) {
        m_ofi_call(fi_close(&(*srx)->fid));
    }
    // always close the EP
    m_ofi_call(fi_close(&(*ep)->fid));
    return m_success;
}

int ofi_util_av(const int n_addr, struct fid_ep* ep, struct fid_av* av, fi_addr_t** addr) {
    size_t ofi_addr_len = ofi_str_len;
    void* ofi_local_addr = calloc(ofi_addr_len, sizeof(char));
    m_ofi_call(fi_getname(&ep->fid, ofi_local_addr, &ofi_addr_len));

    // allocate the temp buffer if not done already
    void* tmp = calloc(n_addr * ofi_addr_len, sizeof(char));
    m_rmem_call(pmi_allgather(ofi_addr_len, ofi_local_addr, &tmp));

    // insert the received addresses
    (*addr) = calloc(n_addr, sizeof(fi_addr_t));
    m_ofi_call(fi_av_insert(av, tmp, n_addr, *addr, 0, NULL));
    for (int i = 0; i < n_addr; ++i) {
        size_t ofi_buf_len = ofi_str_len;
        char name_buf[ofi_str_len];
        fi_av_straddr(av, ((uint8_t*)tmp) + i * ofi_addr_len, name_buf, &ofi_buf_len);
        m_verb("address of rank %d = %s", i, name_buf);
    }
    free(tmp);
    free(ofi_local_addr);

    return m_success;
}

int ofi_util_mr_reg(void* buf, size_t count, uint64_t access, ofi_comm_t* comm,
                         struct fid_mr** mr, void** desc, uint64_t** base_list) {
    m_assert(mr, "mr cannot be NULL");
    //----------------------------------------------------------------------------------------------
    bool useless = false;
    // don't register if the buffer is NULL or the count is 0
    useless |= (!buf || count == 0);
    m_verb("memory regitration for %p is useless? %d",buf,useless);
    // don't register if mr_mode is not MR_LOCAL or FI_HMEM is enabled
    useless |= access & (FI_READ | FI_WRITE | FI_SEND | FI_RECV) &&
               !(comm->prov->domain_attr->mr_mode & (FI_MR_LOCAL | FI_MR_HMEM));
    m_verb("memory regitration for %p is useless? %d",buf,useless);
    // don't dd
    // if it's useless, return
    if (useless) {
        m_verb("memory regitration for %p is useless, skip it",buf);
        *mr = NULL;
        if (desc) {
            *desc = NULL;
        }
    } else {
        //------------------------------------------------------------------------------------------
        // actually register the memory
        // get the flag
#if (M_HAVE_CUDA)
        uint64_t flags = FI_HMEM;
#else
        uint64_t flags = 0x0;
#endif
#if (M_SYNC_RMA_EVENT)
        if (access & (FI_REMOTE_READ | FI_REMOTE_WRITE)) {
            m_verb("using FI_RMA_EVENT to register the MR");
            m_assert(comm->prov->caps & FI_RMA_EVENT, "the provider must have FI_RMA_EVENT");
            flags |= FI_RMA_EVENT;
        }
#endif
        // get device and iface
#if (M_HAVE_CUDA)
        int device;
        enum fi_hmem_iface iface;
        struct cudaPointerAttributes cu_attr;
        m_cuda_call(cudaPointerGetAttributes(&cu_attr, buf));
        switch (cu_attr.type) {
            case (cudaMemoryTypeUnregistered):
                device = 0;
                iface = FI_HMEM_SYSTEM;
                break;
            case (cudaMemoryTypeHost):
                device = cu_attr.device;
                iface = FI_HMEM_CUDA;
                break;
            case (cudaMemoryTypeDevice):
                device = cu_attr.device;
                iface = FI_HMEM_CUDA;
                break;
            case (cudaMemoryTypeManaged):
                device = cu_attr.device;
                iface = FI_HMEM_CUDA;
                break;
            default:
                m_assert(0,"unrecognized memory %d",cu_attr.type);
                break;
        };
#else
        int device = 0;
        enum fi_hmem_iface iface = FI_HMEM_SYSTEM;
#endif
        // register
        uint64_t rkey = 0;
        if (!(comm->prov->domain_attr->mr_mode & FI_MR_PROV_KEY)) {
            rkey = comm->unique_mr_key++;
        }
        struct iovec iov = {
            .iov_base = buf,
            .iov_len = count,
        };
        struct fi_mr_attr attr = {
            .mr_iov = &iov,
            .iov_count = 1,
            .access = access,
            .offset = 0,
            .requested_key =rkey,
            .context = NULL,
            .iface = iface,
            .device = device,
        };
        m_verb("registering memory: key = %llu, flags & FI_RMA_EVENT? %d", rkey,
               (flags & FI_RMA_EVENT) > 0);
        m_ofi_call(fi_mr_regattr(comm->domain, &attr, flags, mr));

        //------------------------------------------------------------------------------------------
        // get the description
        if (access & (FI_READ | FI_WRITE | FI_SEND | FI_RECV)) {
            m_assert(desc, "desc should not be NULL");
            // needed if MR_LOCAL or if HMEM and it's GPU memory
            if ((comm->prov->domain_attr->mr_mode & FI_MR_LOCAL) ||
                (comm->prov->domain_attr->mr_mode & FI_MR_HMEM && iface != FI_HMEM_SYSTEM)) {
                *desc = fi_mr_desc(*mr);
            } else {
                *desc = NULL;
            }
        } else {
            m_assert(!desc, "desc should be NULL");
        }
    }

    //----------------------------------------------------------------------------------------------
    // get the base list, is needed even if we register NULL
    if (access & (FI_REMOTE_READ | FI_REMOTE_WRITE)) {
        m_assert(base_list, "base_list should NOT be null");
        void* list = calloc(ofi_get_size(comm), sizeof(fi_addr_t));
        if (comm->prov->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
            m_verb("fill the base_list");
            m_assert(base_list, "base_list cannot be NULL");
            fi_addr_t usr_base = (fi_addr_t)buf;
            pmi_allgather(sizeof(fi_addr_t), &usr_base, &list);
        }
        *base_list = list;
        m_verb("assign the base_list");
#ifndef NDEBUG
        for (int i = 0; i < ofi_get_size(comm); ++i) {
            m_verb("base[%d] = %llu", i, (*base_list)[i]);
        }
#endif
    } else {
        m_verb("NO base_list");
        m_assert(!base_list, "base list should be NULL");
    }
    return m_success;
}

/**
 * @brief bind the counter to the mr (if not NULL), and bind the MR to the EP (if not null)
*/
int ofi_util_mr_bind(struct fid_ep* ep, struct fid_mr* mr, struct fid_cntr* cntr,
                          ofi_comm_t* comm) {
    if (mr) {
        if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            m_verb("MR_ENDPOINT:");
            // bind the counter to the mr
            if (cntr) {
                m_verb("bind the counter to the MR");
                m_ofi_call(fi_mr_bind(mr, &cntr->fid, FI_REMOTE_WRITE));
            }
            // bind the mr to the ep
            if (ep) {
                uint64_t mr_trx_flags = 0;
                m_verb("bind the MR to the EP");
                m_ofi_call(fi_mr_bind(mr, &ep->fid, mr_trx_flags));
            }
        } else {
            // bind the counter to the EP
            if (cntr && ep) {
                m_verb("no MR_ENDPOINT: bind the counter to the EP directly");
                m_ofi_call(fi_ep_bind(ep, &cntr->fid, FI_REMOTE_WRITE));
            }
        }
    }
    return m_success;
}

int ofi_util_mr_enable(struct fid_mr* mr, ofi_comm_t* comm, uint64_t** key_list) {
    uint64_t key = FI_KEY_NOTAVAIL;
    if (mr && (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT ||
        comm->prov->domain_attr->mr_mode & FI_MR_RMA_EVENT)) {
        m_ofi_call(fi_mr_enable(mr));
    }
    if (key_list) {
        if (mr) {
            key = fi_mr_key(mr);
            m_assert(key != FI_KEY_NOTAVAIL, "the key registration failed");
        }
        void* list = calloc(ofi_get_size(comm), sizeof(uint64_t));
        pmi_allgather(sizeof(key), &key, &list);
        *key_list = (uint64_t*)list;
    }
    return m_success;
}

int ofi_util_mr_close(struct fid_mr* mr){
    if(mr){
        m_ofi_call(fi_close(&mr->fid));
    }
    return m_success;
}

// end of file
