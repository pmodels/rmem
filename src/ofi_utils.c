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

#define ofi_cap_mode     FI_MSG | FI_TAGGED | FI_RMA
#define ofi_cap_ops_tx   FI_READ | FI_WRITE | FI_SEND
#define ofi_cap_ops_rx   FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RECV | FI_RMA_EVENT | FI_DIRECTED_RECV

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
    hints->mode = ~0;
	hints->domain_attr->mode = ~0;
	hints->domain_attr->mr_mode = ~(FI_MR_BASIC | FI_MR_SCALABLE);

    // hint and best_prov bothe evolve as we add capabilities. "hints" is used to test the
    // capability, while best_prov stores them if they match a provider set the minimal requirements
    m_ofi_fatal_info(hints, domain_attr->mr_mode, FI_MR_PROV_KEY);
    m_ofi_fatal_info(hints, caps, FI_RMA | FI_RMA_EVENT);  // implies (REMOTE_)READ/WRITE
    m_ofi_fatal_info(hints, caps, FI_ATOMIC);              // implies (REMOTE_)READ/WRITE
    m_ofi_fatal_info(hints, caps, FI_MSG | FI_TAGGED | FI_DIRECTED_RECV);  // implies SEND/RECV

    // try to get more specific behavior
    m_ofi_test_info(hints, ep_attr->type, FI_EP_RDM);
    m_ofi_test_info(hints, ep_attr->tx_ctx_cnt, FI_SHARED_CONTEXT);
    m_ofi_test_info(hints, ep_attr->rx_ctx_cnt, FI_SHARED_CONTEXT);
    m_ofi_test_info(hints, domain_attr->resource_mgmt, FI_RM_ENABLED);
    m_ofi_test_info(hints, tx_attr->msg_order, FI_ORDER_NONE);
    m_ofi_test_info(hints, rx_attr->msg_order, FI_ORDER_NONE);
    m_ofi_test_info(hints, tx_attr->comp_order, FI_ORDER_NONE);
    m_ofi_test_info(hints, rx_attr->comp_order, FI_ORDER_NONE);

    // check the mode arguments now
    m_ofi_call(fi_getinfo(ofi_ver, NULL, NULL, 0ULL, hints, prov));
    m_assert(*prov, "The provider list is empty");
    m_assert(!((*prov)->mode & FI_RX_CQ_DATA), "need to use FI_MR_RAW");
    m_assert(!((*prov)->mode & FI_ASYNC_IOV), "need to use FI_ASYNC_IOV");
    m_assert(!((*prov)->domain_attr->mr_mode & FI_MR_RAW), "need to use FI_MR_RAW");
    m_assert(!((*prov)->domain_attr->mr_mode & FI_MR_LOCAL), "need to use FI_MR_LOCAL");
    //m_assert((*prov)->tx_attr->inject_size >= OFI_INJECT_THRESHOLD,
    //         "the inject size = %ld must be >= threshold = %d", (*prov)->tx_attr->inject_size,
    //         OFI_INJECT_THRESHOLD);
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
                .caps = ofi_cap_mode | ofi_cap_ops_tx,
                .msg_order = FI_ORDER_NONE,
                .comp_order = FI_ORDER_NONE,
            };
            m_ofi_call(fi_stx_context(dom, &tx_attr, stx, NULL));
            struct fi_rx_attr rx_attr = {
                .caps = ofi_cap_mode | ofi_cap_ops_rx,
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

// end of file
