/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>

#include "ofi.h"
#include "ofi_utils.h"
#include "pmi_utils.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_rma.h"
#include "rmem_utils.h"


#define m_get_rx(i,mem) (i%mem->ofi.n_rx)

int ofi_rmem_init(ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(!(comm->prov->mode &FI_RX_CQ_DATA),"provider needs FI_RX_CQ_DATA");

    //---------------------------------------------------------------------------------------------
    // reset two atomics for the signals with remote write access only
    atomic_store(mem->ofi.epoch + 0, 0);
    atomic_store(mem->ofi.epoch + 1, 0);
    atomic_store(mem->ofi.epoch + 2, 0);

    // allocate the counters tracking the number of issued calls
    mem->ofi.icntr = calloc(comm->size, sizeof(atomic_int));
    for (int i = 0; i < comm->size; ++i) {
        atomic_store(mem->ofi.icntr + i, 0);
    }

    //---------------------------------------------------------------------------------------------
    // register the memory given by the user
    // if we use RMA_INJECT_WRITE we need a valid registration key even if the memory is NULL and
    // the count is 0
#if (OFI_RMA_SYNC_INJECT_WRITE == OFI_RMA_SYNC)
    struct iovec iov = {
        .iov_base = (mem->buf) ? mem->buf : &mem->ofi.tmp,
        .iov_len = (mem->count) ? mem->count : 1,
    };
#elif (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    if (mem->count > 0) {
        struct iovec iov = {
            .iov_base = mem->buf,
            .iov_len = mem->count,
        };
#endif
    struct fi_mr_attr mr_attr = {
        .mr_iov = &iov,
        .iov_count = 1,
        .access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
        .offset = 0,
        .requested_key = 0,
        .context = NULL,
    };
    uint64_t flags = 0;
    m_ofi_call(fi_mr_regattr(comm->domain, &mr_attr, flags, &mem->ofi.mr));
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    } else {
        mem->ofi.mr = NULL;
    }
#endif

    //---------------------------------------------------------------------------------------------
    // allocate one Tx/Rx endpoint per thread context, they all share the transmit queue of the thread
    // the first n_rx endpoints will be Transmit and Receive, the rest is Transmit only
    mem->ofi.n_rx = 1;
    mem->ofi.n_tx = comm->n_ctx;
    m_assert(mem->ofi.n_rx <= mem->ofi.n_tx,"number of rx must be <= number of tx");
    mem->ofi.trx = calloc(comm->n_ctx, sizeof(ofi_rma_trx_t));
    for (int i = 0; i < mem->ofi.n_tx; ++i) {
        // ------------------- endpoint
        if (i < mem->ofi.n_rx) {
            // locally copy the srx address, might be overwritten if needed
            mem->ofi.trx[i].srx = comm->ctx[i].srx;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &mem->ofi.trx[i].ep,
                                         &comm->ctx[i].stx, &mem->ofi.trx[i].srx));
        } else {
            mem->ofi.trx[i].srx = NULL;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &mem->ofi.trx[i].ep,
                                         &comm->ctx[i].stx, &mem->ofi.trx[i].srx));
        }

        // ------------------- address vector
        if (i < mem->ofi.n_rx) {
            m_verb("creating a new AV");
            // if we create a receive context as well, then get the AV
            struct fi_av_attr av_attr = {
                .type = FI_AV_TABLE,
                .name = NULL,
                .count = comm->size,
            };
            m_ofi_call(fi_av_open(comm->domain, &av_attr, &mem->ofi.trx[i].av, NULL));
        }
        m_verb("binding EP #%d to AV %d", i, m_get_rx(i, mem));
        // bind the AV from the corresponding Receive endpoint, otherwise we cannot use it on it
        // because we first build the receive context we are certaint that the AV exists
        m_ofi_call(fi_ep_bind(mem->ofi.trx[i].ep, &mem->ofi.trx[m_get_rx(i, mem)].av->fid, 0));

        // ------------------- counters
        struct fi_cntr_attr rx_cntr_attr = {
            .events = FI_CNTR_EVENTS_COMP,
        };
        m_ofi_call(fi_cntr_open(comm->domain, &rx_cntr_attr, &mem->ofi.trx[i].ccntr, NULL));
        m_ofi_call(fi_cntr_open(comm->domain, &rx_cntr_attr, &mem->ofi.trx[i].rcntr, NULL));
        // completed counter
        uint64_t ccntr_flag = FI_WRITE | FI_READ;
        m_ofi_call(fi_ep_bind(mem->ofi.trx[i].ep, &mem->ofi.trx[i].ccntr->fid, ccntr_flag));
        m_ofi_call(fi_cntr_set(mem->ofi.trx[i].ccntr,0));
        // remote counters
        uint64_t rcntr_flag = FI_REMOTE_WRITE | FI_REMOTE_READ;
        m_ofi_call(fi_ep_bind(mem->ofi.trx[i].ep, &mem->ofi.trx[i].rcntr->fid, rcntr_flag));
        m_ofi_call(fi_cntr_set(mem->ofi.trx[i].rcntr,0));

        // ------------------- completion queue
        struct fi_cq_attr cq_attr = {
            // need to be able to recover the data for PSCW
            .format = OFI_CQ_FORMAT,
        };
        m_ofi_call(fi_cq_open(comm->domain, &cq_attr, &mem->ofi.trx[i].cq, NULL));
        uint64_t tcq_trx_flags = FI_TRANSMIT | FI_RECV;
        m_ofi_call(fi_ep_bind(mem->ofi.trx[i].ep, &mem->ofi.trx[i].cq->fid, tcq_trx_flags));

        // ------------------- finalize
        // enable the EP
        m_ofi_call(fi_enable(mem->ofi.trx[i].ep));
        if (i < mem->ofi.n_rx) {
            // get the addresses from others
            m_rmem_call(ofi_util_av(comm->size, mem->ofi.trx[i].ep, mem->ofi.trx[i].av,
                                     &mem->ofi.trx[i].addr));
            // bind the memory registration
            if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
                && mem->count > 0
#endif
            ) {
                uint64_t mr_trx_flags = 0;
                m_ofi_call(fi_mr_bind(mem->ofi.mr, &mem->ofi.trx[i].ep->fid, mr_trx_flags));
            }
        }
        m_verb("done with EP # %d", i);
    }

    //---------------------------------------------------------------------------------------------
    // if needed, enable the MR and then get the corresponding key and share it
    // obtain the key
    uint64_t key = FI_KEY_NOTAVAIL;
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    if (mem->count > 0) {
#endif
        if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            m_ofi_call(fi_mr_enable(mem->ofi.mr));
        }
        key = fi_mr_key(mem->ofi.mr);
        m_assert(key != FI_KEY_NOTAVAIL, "the key registration failed");
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    }
#endif
    void* key_list = calloc(ofi_get_size(comm), sizeof(uint64_t));
    pmi_allgather(sizeof(key), &key, &key_list);
    mem->ofi.key_list = (uint64_t*)key_list;

    //---------------------------------------------------------------------------------------------
    // if needed allocate sync data structures
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    mem->ofi.sync = calloc(comm->size, sizeof(ofi_cq_t));
    for (int i = 0; i < comm->size; ++i) {
        ofi_cq_t* ccq = mem->ofi.sync + i;
        ccq->kind = m_ofi_cq_kind_sync;
        ccq->cq = mem->ofi.trx[0].cq;
        ccq->sync.cntr = mem->ofi.epoch;
    }
#endif
    return m_success;
    }
int ofi_rmem_free(ofi_rmem_t* mem, ofi_comm_t* comm) {
    struct fid_ep* nullsrx = NULL;
    struct fid_stx* nullstx = NULL;
    // free the Tx first, need to close them before closing the AV in the Rx
    for (int i = 0; i < comm->n_ctx; ++i) {
        m_rmem_call(ofi_util_free_ep(comm->prov, &mem->ofi.trx[i].ep, &nullstx, &nullsrx));
        m_ofi_call(fi_close(&mem->ofi.trx[i].ccntr->fid));
        m_ofi_call(fi_close(&mem->ofi.trx[i].rcntr->fid));
        m_ofi_call(fi_close(&mem->ofi.trx[i].cq->fid));
        if (i < mem->ofi.n_rx) {
            m_ofi_call(fi_close(&mem->ofi.trx[i].av->fid));
            free(mem->ofi.trx[i].addr);
        }
    }
    free(mem->ofi.trx);
    // free the MR
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    if (mem->count > 0) {
#endif
        m_ofi_call(fi_close(&mem->ofi.mr->fid));
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    }
#endif
    free(mem->ofi.key_list);
    // free the counters
    free(mem->ofi.icntr);
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    free(mem->ofi.sync);
#endif
    return m_success;
}

int ofi_rma_init(ofi_rma_t* rma, ofi_rmem_t* pmem, ofi_comm_t* comm) {
    // initialize the context
    rma->ofi.cq.kind = m_ofi_cq_kind_rqst;

    // local IOV
    rma->ofi.iov = (struct iovec){
        .iov_base = rma->buf,
        .iov_len = rma->count,
    };
    // remote IOV
    rma->ofi.riov = (struct fi_rma_iov){
        .addr = rma->disp,  // offset starting from the key registration (FI_MR_SCALABLE)
        .len = rma->count,  // size of the msg
        .key = pmem->ofi.key_list[rma->peer],  // accessing key
    };
    m_assert(rma->ofi.riov.key != FI_KEY_NOTAVAIL, "key must be >0");
    rma->ofi.msg = (struct fi_msg_rma){
        .msg_iov = &rma->ofi.iov,
        .desc = NULL,
        .iov_count = 1,
        .addr = 0,  // set later, once the context ID is known
        .rma_iov = &rma->ofi.riov,
        .rma_iov_count = 1,
        .context = &rma->ofi.cq.ctx,
        .data = 0,
    };
    return m_success;
}

typedef enum {
    RMA_OPT_PUT,
    RMA_OPT_RPUT,
} rma_opt_t;

static int ofi_rma_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id,
                           ofi_comm_t* comm, rma_opt_t op) {
    m_assert(ctx_id < comm->n_ctx, "ctx id = %d < the number of ctx = %d", ctx_id, comm->n_ctx);

    // increment the global counter tracking the issued calls
    atomic_fetch_add(&pmem->ofi.icntr[put->peer], 1);

    // set the completion parameters
    put->ofi.cq.cq = pmem->ofi.trx[ctx_id].cq;

    // address depends on the communicator context
    const int rx_id = m_get_rx(ctx_id,pmem);
    put->ofi.msg.addr =pmem->ofi.trx[rx_id].addr[put->peer];

    // do we inject or generate a cq?
    const bool do_inject = put->count < OFI_INJECT_THRESHOLD;
    // get the flags to use
    uint64_t flags = FI_INJECT_COMPLETE;
    if (do_inject) {
        flags |= FI_INJECT;
    }

    // issue the operation
    switch (op) {
        case (RMA_OPT_PUT): {
            // no increment of the flag with RPUT
            put->ofi.cq.rqst.flag = NULL;
            m_ofi_call(fi_writemsg(pmem->ofi.trx[ctx_id].ep, &put->ofi.msg, flags));
        } break;
        case (RMA_OPT_RPUT): {
            // increment the flag with RPUT
            put->ofi.cq.rqst.flag = &put->ofi.completed;
            atomic_store(put->ofi.cq.rqst.flag, 0);
            // do the communication
            m_ofi_call(fi_writemsg(pmem->ofi.trx[ctx_id].ep, &put->ofi.msg, FI_COMPLETION | flags));

            // if inject, no CQ entry is generated, so the rput is completed upon exit
            if (do_inject) {
                atomic_fetch_add(put->ofi.cq.rqst.flag, 1);
            }
        } break;
    }
    return m_success;
}

int ofi_rput_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_enqueue(put,pmem,ctx_id,comm,RMA_OPT_RPUT);
}

int ofi_put_enqueue(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_enqueue(put,pmem,ctx_id,comm,RMA_OPT_PUT);
}

int ofi_rma_free(ofi_rma_t* rma) {
    return m_success;
}

// notify the processes in comm of memory exposure epoch
int ofi_rmem_post(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // no call access in my memory can be done before the notification, it's safe to reset the
    // counters involved in the memory exposure: epoch[1:2]
    // do NOT reset the epoch[0], it's already exposed to the world!
    atomic_store(mem->ofi.epoch + 1, 0);
    atomic_store(mem->ofi.epoch + 2, 0);

    uint64_t data = m_ofi_data_set_post;
    // notify readiness to the rank list
    for (int i = 0; i < nrank; ++i) {
#if (OFI_RMA_SYNC_INJECT_WRITE == OFI_RMA_SYNC)
        // inject prevents the completion counter to be incremented
        // WARNING: even if the key is unused, not giving it doesn't work
        uint64_t disp = 0;
        fi_addr_t* dest_addr = mem->ofi.trx[0].addr + rank[i];
        uint64_t key = mem->ofi.key_list[rank[i]];
        m_assert(key != FI_KEY_NOTAVAIL, "key must be valid");
        // use any local buffer, we write 0-byte anyway
        m_ofi_call(
            fi_inject_writedata(mem->ofi.trx[0].ep, &mem->ofi.tmp, 0, data, *dest_addr, disp, key));
#elif (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
        //m_ofi_call(fi_inject(mem->ofi.trx[0].ep, &data, 8, mem->ofi.trx[0].addr[rank[i]]));
        uint64_t tag = m_ofi_tag_sync;
        m_ofi_call(fi_tinject(mem->ofi.trx[0].ep, &data, sizeof(data),
                              mem->ofi.trx[0].addr[rank[i]], tag));
#endif
    }
    // inject_write data will generate a counter update, need to take that into account. we separate
    // the loops to not penalize the other waiting processes
    for (int i = 0; i < nrank; ++i) {
        atomic_fetch_add(&mem->ofi.icntr[rank[i]], 1);
    }
    return m_success;
}

// wait for the processes in comm to notify their exposure
int ofi_rmem_start(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // store the flag pointers and the cq to progress
    ofi_cq_t cq = {
        .kind = m_ofi_cq_kind_sync,
        .sync.cntr = mem->ofi.epoch,
        .cq = mem->ofi.trx[0].cq,
    };
#if (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    for (int i = 0; i < nrank; ++i) {
        uint64_t ignore = 0x0;
        uint64_t tag = m_ofi_tag_sync;
        m_ofi_call(fi_trecv(mem->ofi.trx[0].srx, &mem->ofi.sync[i].sync.data, sizeof(uint64_t),
                            NULL, mem->ofi.trx[0].addr[rank[i]], tag, ignore,
                            &mem->ofi.sync[i].ctx));
    }
#endif
    while (atomic_load(mem->ofi.epoch + 0) < nrank) {
        // trigger progress to change the values of the epoch
        ofi_progress(&cq);
    }
    // once we have received everybody's signal, resets epoch[0] for the next iteration
    // nobody can post until I have completed on my side, so it will no lead to data race
    atomic_store(mem->ofi.epoch + 0, 0);
    return m_success;
}

int ofi_rmem_complete(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // store the flag pointers and the cq to progress
    ofi_cq_t cq = {
        .kind = m_ofi_cq_kind_sync,
        .sync.cntr = mem->ofi.epoch,
        .cq = mem->ofi.trx[0].cq,
    };

    //----------------------------------------------------------------------------------------------
    // count the number of calls issued for each of the ranks and notify them
    int ttl_issued = 0;
    for (int i = 0; i < nrank; ++i) {
        int issued_rank = atomic_exchange(&mem->ofi.icntr[rank[i]], 0);
        ttl_issued += issued_rank;

        // notify
#if (OFI_RMA_SYNC_INJECT_WRITE == OFI_RMA_SYNC)
        uint64_t disp = 0;
        fi_addr_t* dest_addr = mem->ofi.trx[0].addr + rank[i];
        uint64_t key = mem->ofi.key_list[rank[i]];
        uint64_t data = m_ofi_data_set_cmpl | m_ofi_data_set_nops(issued_rank);
        m_ofi_call(
            fi_inject_writedata(mem->ofi.trx[0].ep, &mem->ofi.tmp, 0, data, *dest_addr, disp, key));
#elif (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
        uint64_t data = m_ofi_data_set_cmpl | m_ofi_data_set_nops(issued_rank);
        // m_ofi_call(fi_inject(mem->ofi.trx[0].ep, &data, 8, mem->ofi.trx[0].addr[rank[i]]));
        uint64_t tag = m_ofi_tag_sync;
        m_ofi_call(fi_tinject(mem->ofi.trx[0].ep, &data, sizeof(data),
                              mem->ofi.trx[0].addr[rank[i]], tag));
#endif
    }
    //----------------------------------------------------------------------------------------------
    // count the number of completed calls and wait till they are all done
    // must complete all the sync call done in rmem_post (if any) + the sync call done with RMA
#if (OFI_RMA_SYNC_INJECT_WRITE == OFI_RMA_SYNC)
    int rma_sync_count = nrank;
#elif (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    // no rma calls have been made, none to take into account in the remote cntr
    int rma_sync_count = 0;
#endif
    uint64_t threshold = ttl_issued + rma_sync_count;
    uint64_t ttl_completed = 0;
    if (comm->n_ctx == 1) {
        fi_cntr_wait(mem->ofi.trx[0].ccntr, threshold, -1);
        fi_cntr_set(mem->ofi.trx[0].ccntr, 0);
        ttl_completed = threshold;
    } else {
        while (ttl_completed < threshold ) {
            // force progress on the trx[0] cq to make sure we increment the completion counters for
            // providers where progress is not automatic
            if (comm->prov->domain_attr->control_progress & FI_PROGRESS_MANUAL) {
                ofi_progress(&cq);
            }
            for (int i = 0; i < comm->n_ctx; ++i) {
                int nc = fi_cntr_read(mem->ofi.trx[i].ccntr);
                if (nc > 0) {
                    ttl_completed += nc;
                    m_ofi_call(fi_cntr_add(mem->ofi.trx[i].ccntr, (~nc + 0x1)));
                }
                // if the new value makes the value match, break
                if (ttl_completed >= ttl_issued) {
                    break;
                }
            }
        }
    }
    m_assert(ttl_completed == (ttl_issued + rma_sync_count), "ttl_completed = %" PRIu64 ", ttl_issued = %d",
             ttl_completed, ttl_issued + rma_sync_count);
    return m_success;
}
int ofi_rmem_wait(const int nrank, const int* rank, ofi_rmem_t* mem, ofi_comm_t* comm) {
    // store the flag pointers and the cq to progress
    ofi_cq_t cq = {
        .kind = m_ofi_cq_kind_sync,
        .sync.cntr = mem->ofi.epoch,
        .cq = mem->ofi.trx[0].cq,
    };

#if (OFI_RMA_SYNC_INJECT_WRITE == OFI_RMA_SYNC)
    int rma_sync_count = nrank;
#elif (OFI_RMA_SYNC_MSG == OFI_RMA_SYNC)
    for (int i = 0; i < nrank; ++i) {
        uint64_t ignore = 0x0;
        uint64_t tag = m_ofi_tag_sync;
        m_ofi_call(fi_trecv(mem->ofi.trx[0].srx, &mem->ofi.sync[i].sync.data, sizeof(uint64_t),
                            NULL, mem->ofi.trx[0].addr[rank[i]], tag, ignore,
                            &mem->ofi.sync[i].ctx));
    }
    // no rma calls have been made, none to take into account in the remote cntr
    int rma_sync_count = 0;
#endif

    // compare the number of calls done to the value in the epoch if everybody has finished
    // n_rcompleted must be = the total number of calls received (including during the start sync) +
    // the sync from nrank for this sync
    if (mem->ofi.n_rx == 1) {
        while (atomic_load(mem->ofi.epoch + 1) < nrank) {
            ofi_progress(&cq);
        }
        uint64_t threshold = atomic_load(mem->ofi.epoch + 2) + rma_sync_count;
        fi_cntr_wait(mem->ofi.trx[0].rcntr, threshold, -1);
        fi_cntr_set(mem->ofi.trx[0].rcntr, 0);
    } else {
        uint64_t n_rcompleted = 0;
        while (mem->ofi.epoch[1] < nrank ||
               n_rcompleted < (atomic_load(mem->ofi.epoch + 2) + rma_sync_count)) {
            for (int i = 0; i < mem->ofi.n_rx; ++i) {
                // count the number of remote calls over the receive contexts
                uint64_t n_ri = fi_cntr_read(mem->ofi.trx[i].rcntr);
                if (n_ri > 0) {
                    n_rcompleted += n_ri;
                    // substract them form the counter as they have been taken into account must
                    // offset the remote completion counter, don't reset it to 0 as someone might
                    // already issue RMA calls to my memory as part of their post
                    m_ofi_call(fi_cntr_add(mem->ofi.trx[i].rcntr, (~n_ri + 0x1)));
                }
            }
            // run progress to update the epoch counters
            ofi_progress(&cq);
        }
    }
    return m_success;
}

// end of file
