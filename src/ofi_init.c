/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include <stdlib.h>

#include "ofi.h"
#include "ofi_utils.h"
#include "pmi_utils.h"
#include "rdma/fabric.h"
#include "rdma/fi_domain.h"
#include "rmem_utils.h"


// create a communication context
int ofi_ctx_init(const int comm_size, struct fi_info* prov, struct fid_domain* dom,
                 ofi_ctx_t** ctx) {
    // // create the new EP
    (*ctx)->srx = NULL;
    (*ctx)->stx = NULL;
    m_rmem_call(ofi_util_new_ep(true,prov, dom, &(*ctx)->p2p_ep, &(*ctx)->stx, &(*ctx)->srx));

    // open a cq queue
    struct fi_cq_attr cq_attr = {
        .format = OFI_CQ_FORMAT,
    };
    m_ofi_call(fi_cq_open(dom, &cq_attr, &(*ctx)->p2p_cq, NULL));

    // associate it to the endpoint for inbound and outbound completion
    // add selective completion to NOT have a completion unless requested
    // WARNING: SELECTIVE_COMPLETION is more expensive than discarding the cq entry
    uint64_t tag = FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION;
    m_ofi_call(fi_ep_bind((*ctx)->p2p_ep, &(*ctx)->p2p_cq->fid, tag));

    // associate the adress vector
    struct fi_av_attr av_attr = {
        .type = FI_AV_TABLE,
        .name = NULL,
        .count = comm_size,
    };
    m_ofi_call(fi_av_open(dom, &av_attr, &(*ctx)->p2p_av, NULL));
    m_ofi_call(fi_ep_bind((*ctx)->p2p_ep, &(*ctx)->p2p_av->fid, 0));

    // enable the endpoint
    m_ofi_call(fi_enable((*ctx)->p2p_ep));

    //----------------------------------------------------------------------------------------------
    // fill the address vector, needs an ep to be connected
    // every endpoint has a different address vector. the memory belongs to the domain, but the
    // values are endpoint dependent open the address vector
    // get the address from libfabric and use the PMI to communicate it to others
    m_rmem_call(ofi_util_av(comm_size, (*ctx)->p2p_ep, (*ctx)->p2p_av, &(*ctx)->p2p_addr));

    return m_success;
}

int ofi_ctx_free(struct fi_info* prov, ofi_ctx_t** ctx){
    free((*ctx)->p2p_addr);
    m_rmem_call(ofi_util_free_ep(prov,&(*ctx)->p2p_ep, &(*ctx)->stx, &(*ctx)->srx));
    m_ofi_call(fi_close(&(*ctx)->p2p_cq->fid));
    m_ofi_call(fi_close(&(*ctx)->p2p_av->fid));
    return m_success;
}

// main functions
int ofi_init(ofi_comm_t* ofi) {

    // get the provider list
    m_rmem_call(ofi_util_get_prov(&ofi->prov));
    // struct fi_info* prov_list;
    // m_ofi_call(fi_getinfo(ofi_ver, NULL, NULL, 0ULL, NULL, &prov_list));
    //
    // struct fi_info* hints = fi_allocinfo();
    // m_assert(hints, "failure in fi_allocinfo");
    // m_rmem_call(ofi_set_info(hints,true));
    //
    // // get the infos, returns a list of providers
    // m_ofi_call(fi_getinfo(ofi_ver, NULL, NULL, 0ULL, hints, &ofi->prov));
    // m_assert(ofi->prov, "failure to find a provider");
    // fi_freeinfo(hints);

    // make sure a few options are set
    // m_rmem_call(ofi_set_info(ofi->prov,false));

    // open the fabric + domain for an obtained provider
    m_ofi_call(fi_fabric(ofi->prov->fabric_attr, &ofi->fabric, NULL));
    m_ofi_call(fi_domain(ofi->fabric, ofi->prov, &ofi->domain, NULL));

    // get the comm rank and comm_size
    m_rmem_call(pmi_init());
    m_rmem_call(pmi_get_comm_id(&ofi->rank, &ofi->size));

    // set the initial value of unique_key
    ofi->unique_mr_key = 0;
    //----------------------------------------------------------------------------------------------
    // open different communication contexts
    ofi->ctx = calloc(ofi->n_ctx, sizeof(ofi_ctx_t));
    for (int i = 0; i < ofi->n_ctx; ++i) {
        ofi_ctx_t* cctx = ofi->ctx + i;
        m_rmem_call(ofi_ctx_init(ofi->size, ofi->prov, ofi->domain, &cctx));
    }

    return m_success;
}
int ofi_finalize(ofi_comm_t* ofi) {
    for (int i = 0; i < ofi->n_ctx; ++i) {
        ofi_ctx_t* cctx = ofi->ctx + i;
        ofi_ctx_free(ofi->prov, &cctx);
    }
    free(ofi->ctx);
    m_ofi_call(fi_close(&ofi->domain->fid));
    m_ofi_call(fi_close(&ofi->fabric->fid));
    fi_freeinfo(ofi->prov);
    m_rmem_call(pmi_finalize());

    return m_success;
}
