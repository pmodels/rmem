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
                hint->field &= ~(value);                                  \
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
                m_verb("successfully set " #field " to " #value);         \
            }                                                             \
        } else {                                                          \
            m_verb(#field " already has " #value);                        \
        }                                                                 \
    } while (0)

typedef enum {
    M_OFI_EP_KIND_NORMAL,
    M_OFI_EP_KIND_SHARED,
} ofi_ep_kind_t;

static int ofi_prov_score(char* provname, ofi_cap_t* caps) {
    if (0 == strcmp(provname, "cxi")) {
        if (caps) {
            *caps = M_OFI_PROV_HAS_RMA_EVENT | M_OFI_PROV_HAS_ATOMIC | M_OFI_PROV_HAS_FENCE;
        }
        return 3;
    } else if (0 == strcmp(provname, "verbs;ofi_rxm")) {
        if (caps) {
            *caps = M_OFI_PROV_HAS_ATOMIC | M_OFI_PROV_HAS_CQ_DATA
#if (!M_FORCE_MR_LOCAL)
                    | M_OFI_PROV_HAS_ATOMIC
#endif
                ;
        }
        return 2;
    } else if (0 == strcmp(provname, "psm3")) {
        if (caps) {
            *caps = M_OFI_PROV_HAS_RMA_EVENT | M_OFI_PROV_HAS_ATOMIC | M_OFI_PROV_HAS_CQ_DATA;
        }
        return 1;
    } else if (0 == strcmp(provname, "sockets")) {
        if (caps) {
            *caps = M_OFI_PROV_HAS_ATOMIC | M_OFI_PROV_HAS_CQ_DATA;
        }
        return 1;
    } else if (0 == strcmp(provname, "tcp;ofi_rxm")) {
        if (caps) {
            *caps = M_OFI_PROV_HAS_CQ_DATA
#if (!M_FORCE_MR_LOCAL)
                    | M_OFI_PROV_HAS_ATOMIC
#endif
                ;
        }
        return 1;
    } else if (0 == strcmp(provname, "tcp")) {
        if (caps) {
            *caps = M_OFI_PROV_HAS_CQ_DATA;
        }
        return 1;
    }
    return 0;
}

/**
 * @brief given a provider's capability set, determine the best modes to use
 */
static int ofi_prov_mode(ofi_cap_t* prov_cap, ofi_mode_t* mode, uint64_t* ofi_cap) {
    //----------------------------------------------------------------------------------------------
    // [1] ready-to-receive
    if (mode->rtr_mode) {
        switch (mode->rtr_mode) {
            case (M_OFI_RTR_NULL):
                m_assert(0, "null is not supported here");
                break;
            case M_OFI_RTR_MSG:
                m_verb("PROV MODE - RTR: doing msgs");
                *ofi_cap |= FI_MSG;
                break;
            case M_OFI_RTR_ATOMIC:
                m_verb("PROV MODE - RTR: doing atomic");
                *ofi_cap |= FI_ATOMIC;
                m_assert(m_ofi_prov_has_atomic(*prov_cap), "provider needs atomics capabilities");
                break;
            case M_OFI_RTR_TAGGED:
                m_verb("PROV MODE - RTR: doing tagged");
                *ofi_cap |= FI_TAGGED;
                break;
        }
    } else {
        *ofi_cap |= FI_MSG;
        mode->rtr_mode = M_OFI_RTR_MSG;
        m_verb("PROV MODE - RTR: doing msgs");
    }
    //----------------------------------------------------------------------------------------------
    // [4] down-to-close
    if (mode->dtc_mode) {
        switch (mode->dtc_mode) {
            case (M_OFI_RTR_NULL):
                m_assert(0, "null is not supported here");
                break;
            case M_OFI_RTR_MSG:
                m_verb("PROV MODE - DTC: doing msgs");
                *ofi_cap |= FI_MSG;
                break;
            case M_OFI_RTR_TAGGED:
                m_verb("PROV MODE - DTC: doing tagged");
                *ofi_cap |= FI_TAGGED;
                break;
        }
    } else {
        *ofi_cap |= FI_MSG;
        mode->dtc_mode = M_OFI_DTC_MSG;
        m_verb("PROV MODE - DTC: doing msgs");
    }
    //----------------------------------------------------------------------------------------------
    // [2] remote completion
    if (mode->rcmpl_mode) {
        switch (mode->rcmpl_mode) {
            case (M_OFI_RCMPL_NULL):
                m_assert(0, "null is not supported here");
                break;
            case M_OFI_RCMPL_CQ_DATA:
                m_verb("PROV MODE - RCMPL: doing cq data");
                m_assert(m_ofi_prov_has_cq_data(*prov_cap), "provider needs cq data capabilities");
                break;
            case M_OFI_RCMPL_REMOTE_CNTR:
                m_verb("PROV MODE - RCMPL: doing remote counter");
                *ofi_cap |= FI_RMA_EVENT;
                m_assert(m_ofi_prov_has_rma_event(*prov_cap),
                         "provider needs rma event capabilities");
                break;
            case M_OFI_RCMPL_FENCE:
                m_verb("PROV MODE - RCMPL: doing fence");
                *ofi_cap |= FI_FENCE;
                m_assert(m_ofi_prov_has_fence(*prov_cap), "provider needs fence capabilities");
                break;
            case M_OFI_RCMPL_DELIV_COMPL:
                m_verb("PROV MODE - RCMPL: doing delivery complete");
                break;
        }
    } else {
        if (m_ofi_prov_has_cq_data(*prov_cap)) {
            m_verb("PROV MODE - RCMPL: doing cq data");
            mode->rcmpl_mode = M_OFI_RCMPL_CQ_DATA;
        } else if (m_ofi_prov_has_rma_event(*prov_cap)) {
            *ofi_cap |= FI_RMA_EVENT;
            m_verb("PROV MODE - RCMPL: doing remote counter");
            mode->rcmpl_mode = M_OFI_RCMPL_REMOTE_CNTR;
        } else if (m_ofi_prov_has_fence(*prov_cap)) {
            m_verb("PROV MODE - RCMPL: doing fence");
            *ofi_cap |= FI_FENCE;
            mode->rcmpl_mode = M_OFI_RCMPL_FENCE;
        } else {
            m_verb("PROV MODE - RCMPL: doing delivery complete");
            mode->rcmpl_mode = M_OFI_RCMPL_DELIV_COMPL;
        }
    }
    //----------------------------------------------------------------------------------------------
    // [3] signal mode
    if (mode->sig_mode) {
        switch (mode->sig_mode) {
            case (M_OFI_SIG_NULL):
                m_assert(0, "null is not supported here");
                break;
            case M_OFI_SIG_CQ_DATA:
                m_verb("PROV MODE - SIG: doing cq data");
                m_assert(m_ofi_prov_has_cq_data(*prov_cap), "provider needs cq data capabilities");
                break;
            case M_OFI_SIG_ATOMIC:
                m_verb("PROV MODE - SIG: doing atomic + fence");
                *ofi_cap |= FI_ATOMIC | FI_FENCE;
                m_assert(m_ofi_prov_has_atomic(*prov_cap), "provider needs atomics capabilities");
                m_assert(m_ofi_prov_has_fence(*prov_cap), "provider needs fencing capabilities");
                break;
        }
    } else {
        if (m_ofi_prov_has_cq_data(*prov_cap)) {
            m_verb("PROV MODE - SIG: doing cq data");
            mode->sig_mode = M_OFI_SIG_CQ_DATA;
        } else if (m_ofi_prov_has_atomic(*prov_cap) && m_ofi_prov_has_fence(*prov_cap)) {
            m_verb("PROV MODE - SIG: doing atomic + fence");
            *ofi_cap |= FI_ATOMIC | FI_FENCE;
            mode->sig_mode = M_OFI_SIG_ATOMIC;
        } else {
            m_assert(0, "unable to use signals");
        }
    }
    return m_success;
}

int ofi_util_get_prov(struct fi_info** prov, ofi_mode_t* prov_mode) {
    // get the list of available providers and select the best one
    int ofi_ver = fi_version();
    int best_score = -1;
    struct fi_info* best_prov = NULL;
    struct fi_info* prov_list = NULL;
    m_ofi_call(fi_getinfo(ofi_ver, NULL, NULL, 0ULL, NULL, &prov_list));
    for (struct fi_info* cprov = prov_list; cprov; cprov = cprov->next) {
        int score = ofi_prov_score(cprov->fabric_attr->prov_name,NULL);
        if (score > best_score) {
            best_score = score;
            best_prov = cprov;
        }
    }
    // get the set of capabilities
    uint8_t prov_cap = 0x00;
    best_score = ofi_prov_score(best_prov->fabric_attr->prov_name, &prov_cap);
    m_assert(prov_cap, "we need at least a few capabilities (cannot be %d)", prov_cap);
    char* prov_name = best_prov->fabric_attr->prov_name;
    m_verb("best provider is %s", prov_name);

    // get the best provider's name
    struct fi_info* hints = fi_allocinfo();
    hints->fabric_attr->prov_name = malloc(strlen(prov_name) + 1);
    strcpy(hints->fabric_attr->prov_name, prov_name);
    fi_freeinfo(prov_list);  // no need of prov_list anymore

    //----------------------------------------------------------------------------------------------
    // get the operational modes
    uint64_t mycap = FI_MSG | FI_TAGGED | FI_RMA | FI_DIRECTED_RECV;
    m_rmem_call(ofi_prov_mode(&prov_cap, prov_mode, &mycap));

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
#if (M_FORCE_MR_LOCAL)
    hints->domain_attr->mr_mode |= FI_MR_LOCAL;
#endif
    hints->domain_attr->mr_mode |= FI_MR_PROV_KEY;
    hints->domain_attr->mr_mode |= FI_MR_ALLOCATED;
    hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR;

    // Reliable DatagraM (RDM)
    hints->ep_attr->type = FI_EP_RDM;
    // make sure the provider has that
    m_ofi_fatal_info(hints, caps, mycap);

    //----------------------------------------------------------------------------------------------
    // optional
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
#if (M_FORCE_ASYNC_PROGRESS)
    m_ofi_test_info(hints, domain_attr->data_progress, FI_PROGRESS_MANUAL);
    m_ofi_test_info(hints, domain_attr->control_progress, FI_PROGRESS_MANUAL);
#endif
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
    (*prov)->domain_attr->mr_mode |= FI_MR_LOCAL;

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
                .caps = prov->caps,
                .msg_order = FI_ORDER_NONE,
                .comp_order = FI_ORDER_NONE,
            };
            m_ofi_call(fi_stx_context(dom, &tx_attr, stx, NULL));
            struct fi_rx_attr rx_attr = {
                .caps = prov->caps,
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
    m_verb("memory regitration for %p is useless? %d", buf, useless);
    // don't register if mr_mode is not MR_LOCAL or FI_HMEM is enabled
    useless |= access & (FI_READ | FI_WRITE | FI_SEND | FI_RECV) &&
               !(comm->prov->domain_attr->mr_mode & (FI_MR_LOCAL | FI_MR_HMEM));
    m_verb("registering %p (count = %lu): access is right?%llu MR_LOCAL? %d", buf, count,
           access & (FI_READ | FI_WRITE | FI_SEND | FI_RECV),
           comm->prov->domain_attr->mr_mode & (FI_MR_LOCAL | FI_MR_HMEM));
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
        if ((comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_REMOTE_CNTR) &&
            access & (FI_REMOTE_READ | FI_REMOTE_WRITE)) {
            m_verb("using FI_RMA_EVENT to register the MR");
            m_assert(comm->prov->caps & FI_RMA_EVENT, "the provider must have FI_RMA_EVENT");
            flags |= FI_RMA_EVENT;
        }
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
        m_verb("closing memory");
        m_ofi_call(fi_close(&mr->fid));
    }
    return m_success;
}

//--------------------------------------------------------------------------------------------------
int ofi_util_sig_reg(ofi_mem_sig_t* sig, ofi_comm_t* comm) {
    sig->val = 0;
    sig->res = 0;
    sig->inc = 1;
    m_rmem_call(ofi_util_mr_reg(&sig->val, sizeof(uint32_t), FI_REMOTE_READ | FI_REMOTE_WRITE, comm,
                                &sig->val_mr.mr, NULL, &sig->val_mr.base_list));
    m_rmem_call(ofi_util_mr_reg(&sig->inc, sizeof(uint32_t), FI_READ | FI_WRITE, comm,
                                &sig->inc_mr.mr, &sig->inc_mr.desc, NULL));
    m_rmem_call(ofi_util_mr_reg(&sig->res, sizeof(uint32_t), FI_READ | FI_WRITE, comm,
                                &sig->res_mr.mr, &sig->res_mr.desc, NULL));
    return m_success;
}
int ofi_util_sig_bind(ofi_mem_sig_t* sig, struct fid_ep* ep, ofi_comm_t* comm) {
    m_rmem_call(ofi_util_mr_bind(ep, sig->inc_mr.mr, NULL, comm));
    m_rmem_call(ofi_util_mr_bind(ep, sig->val_mr.mr, NULL, comm));
    m_rmem_call(ofi_util_mr_bind(ep, sig->res_mr.mr, NULL, comm));
    return m_success;
}
int ofi_util_sig_enable(ofi_mem_sig_t* sig, ofi_comm_t* comm) {
    m_rmem_call(ofi_util_mr_enable(sig->val_mr.mr, comm, &sig->val_mr.key_list));
    m_rmem_call(ofi_util_mr_enable(sig->inc_mr.mr, comm, NULL));
    m_rmem_call(ofi_util_mr_enable(sig->res_mr.mr, comm, NULL));
    return m_success;
}
int ofi_util_sig_close(ofi_mem_sig_t* sig) {
    m_rmem_call(ofi_util_mr_close(sig->val_mr.mr));
    m_rmem_call(ofi_util_mr_close(sig->inc_mr.mr));
    m_rmem_call(ofi_util_mr_close(sig->res_mr.mr));
    free(sig->val_mr.key_list);
    free(sig->val_mr.base_list);
    return m_success;
}
/**
 * @brief progress till the value exposed in sig has reached the desired threshold
 *
 * WARNING: the rank, addr, EP, and CQ in progress MUST be compatible
*/
int ofi_util_sig_wait(ofi_mem_sig_t* sig, int myrank, fi_addr_t myaddr, struct fid_ep* ep,
                           ofi_progress_t* progress, uint32_t threshold) {
    // setup the data structures
    ofi_cqdata_t* cqdata = &sig->read_cqdata;
    cqdata->kind = m_ofi_cq_kind_rqst;
    m_countr_store(&cqdata->rqst.busy, 0);
    // even if we read it, we need to provide a source buffer
    struct fi_ioc iov = {
        .addr = &sig->inc,
        .count = 1,
    };
    struct fi_ioc res_iov = {
        .addr = &sig->res,
        .count = 1,
    };
    struct fi_rma_ioc rma_iov = {
        .count = 1,
        .addr = sig->val_mr.base_list[myrank] + 0,
        .key = sig->val_mr.key_list[myrank],
    };
    struct fi_msg_atomic msg = {
        .msg_iov = &iov,
        .desc = &sig->inc_mr.desc,
        .iov_count = 1,  // 1,
        .addr = myaddr,  // myself
        .rma_iov = &rma_iov,
        .rma_iov_count = 1,
        .datatype = FI_INT32,
        .op = FI_ATOMIC_READ,
        .data = 0x0,
        .context = &cqdata->ctx,
    };
    int it = 0;
    while (sig->res < threshold) {
        // issue a fi_fetch
        m_verb("issuing an atomic number %d, res = %d", it, sig->res);
        int curr = m_countr_exchange(&sig->read_cqdata.rqst.busy, 1);
        m_assert(!curr, "current value should be 0 and not %d", curr);
        m_ofi_call_again(
            fi_fetch_atomicmsg(ep, &msg, &res_iov, &sig->res_mr.desc, 1, FI_TRANSMIT_COMPLETE),
            progress);
        // count the number of issued atomics
        it++;
        // wait for completion of the atomic
        while (m_countr_load(&sig->read_cqdata.rqst.busy)) {
            ofi_progress(progress);
        }
        m_verb("atomics has completed, res = %d, busy = %d", sig->res,m_countr_load(&sig->read_cqdata.rqst.busy));
    }
    sig->res = 0;
    return m_success;
}

// end of file
