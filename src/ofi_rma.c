/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
#include <inttypes.h>
#include <stdint.h>
#include <unistd.h>

#include "ofi.h"
#include "ofi_rma_sync_tools.h"
#include "ofi_utils.h"
#include "pmi_utils.h"
#include "rdma/fi_atomic.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_rma.h"

#define m_get_rx(i, mem) (i % mem->ofi.n_rx)

int ofi_rmem_init(ofi_rmem_t* mem, ofi_comm_t* comm) {
    m_assert(!(comm->prov->mode & FI_RX_CQ_DATA), "provider needs FI_RX_CQ_DATA");
    //---------------------------------------------------------------------------------------------
    // reset two atomics for the signals with remote write access only
    for (int i = 0; i < m_rma_n_epoch; ++i) {
        m_countr_init(mem->ofi.sync.epch + i);
    }

    // allocate the counters tracking the number of issued calls
    mem->ofi.sync.icntr = calloc(comm->size, sizeof(int));

    //---------------------------------------------------------------------------------------------
    // register the memory given by the user
    m_verb("registering user memory");
    m_rmem_call(ofi_util_mr_reg(mem->buf, mem->count, FI_REMOTE_READ | FI_REMOTE_WRITE, comm,
                                &mem->ofi.mr.mr, NULL, &mem->ofi.mr.base_list));
    m_verb("registered memory -> %p",mem->ofi.mr.mr);

    //----------------------------------------------------------------------------------------------
    // the sync data needed for the Post-start atomic protocol
    if (comm->prov_mode.rtr_mode == M_OFI_RTR_ATOMIC) {
        m_rmem_call(ofi_util_sig_reg(&mem->ofi.sync.ps_sig, comm));
    }
    //---------------------------------------------------------------------------------------------
    // open shared completion and remote counter
    if (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_REMOTE_CNTR) {
        struct fi_cntr_attr cntr_attr = {
            .events = FI_CNTR_EVENTS_COMP,
            .wait_obj = FI_WAIT_UNSPEC,
        };
        // remote counters - count the number of fi_write/fi_read targeted to me
        m_verb("open remote counter");
        m_ofi_call(fi_cntr_open(comm->domain, &cntr_attr, &mem->ofi.rcntr, NULL));
        m_ofi_call(fi_cntr_set(mem->ofi.rcntr, 0));
    }
    //---------------------------------------------------------------------------------------------
    // allocate one Tx/Rx endpoint per thread context, they all share the transmit queue of the
    // thread the first n_rx endpoints will be Transmit and Receive, the rest is Transmit only
    mem->ofi.n_rx = 1;
    mem->ofi.n_tx = comm->n_ctx;
    const int n_ttl_trx = comm->n_ctx + 1;
    m_assert(mem->ofi.n_rx <= mem->ofi.n_tx, "number of rx must be <= number of tx");
    // allocate n_ctx + 1 structs and get the the right pointer ids
    ofi_rma_trx_t* trx = calloc(n_ttl_trx, sizeof(ofi_rma_trx_t));
    mem->ofi.data_trx = trx + 0;
    mem->ofi.sync_trx = trx + comm->n_ctx;
    for (int i = 0; i < n_ttl_trx; ++i) {
        const bool is_rx = (i < mem->ofi.n_rx);
        const bool is_tx = (i < mem->ofi.n_tx);
        const bool is_sync = (i == comm->n_ctx);
        m_verb("-----------------");
        m_verb("creating EP %d/%d: is_rx? %d, is_tx? %d, is_sync? %d", i, n_ttl_trx, is_rx, is_tx,
               is_sync);

        // ------------------- endpoint
        if (is_rx) {
            // locally copy the srx address, might be overwritten if needed
            trx[i].srx = comm->ctx[i].srx;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &trx[i].ep,
                                        &comm->ctx[i].stx, &trx[i].srx));
        } else if (is_tx) {
            mem->ofi.data_trx[i].srx = NULL;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &trx[i].ep,
                                        &comm->ctx[i].stx, &trx[i].srx));
        } else {
            // thread 0 will do the sync
            trx[i].srx = comm->ctx[0].srx;
            m_rmem_call(ofi_util_new_ep(false, comm->prov, comm->domain, &trx[i].ep,
                                        &comm->ctx[0].stx, &trx[i].srx));
        }

        // ------------------- address vector
        if (is_rx || is_sync) {
            m_verb("creating a new AV and binding it");
            // if we create a receive context as well, then get the AV
            struct fi_av_attr av_attr = {
                .type = FI_AV_TABLE,
                .name = NULL,
                .count = comm->size,
            };
            m_ofi_call(fi_av_open(comm->domain, &av_attr, &trx[i].av, NULL));
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].av->fid, 0));
        } else {
            // bind the AV from the corresponding Receive endpoint, otherwise we cannot use it on it
            // because we first build the receive context we are certain that the AV exists
            m_verb("binding EP #%d to AV %d", i, m_get_rx(i, mem));
            m_ofi_call(fi_ep_bind(trx[i].ep, &trx[m_get_rx(i, mem)].av->fid, 0));
        }

        // ------------------- completion queue
        struct fi_cq_attr cq_attr = {
            .format = OFI_CQ_FORMAT,
            .wait_obj = FI_WAIT_NONE,
        };
        m_ofi_call(fi_cq_open(comm->domain, &cq_attr, &trx[i].cq, NULL));
        uint64_t tcq_trx_flags = FI_TRANSMIT | FI_RECV;
        m_ofi_call(fi_ep_bind(trx[i].ep, &trx[i].cq->fid, tcq_trx_flags));

        //------------------- bind the counters and the MR
        // if MR_ENDPOINT we have to enable the EP first and then bind the MR
        if (comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            m_verb("enable the EP");
            m_ofi_call(fi_enable(trx[i].ep));
        }
        if (is_rx) {
            if (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_REMOTE_CNTR) {
                m_rmem_call(ofi_util_mr_bind(trx[i].ep, mem->ofi.mr.mr, mem->ofi.rcntr, comm));
            } else {
                m_rmem_call(ofi_util_mr_bind(trx[i].ep, mem->ofi.mr.mr, NULL, comm));
            }
        }
        if (is_sync && comm->prov_mode.rtr_mode == M_OFI_RTR_ATOMIC) {
            m_rmem_call(ofi_util_sig_bind(&mem->ofi.sync.ps_sig, trx[i].ep, comm));
        }
        // is not MR_ENDPOINT, first bind and then enable the EP
        if (!(comm->prov->domain_attr->mr_mode & FI_MR_ENDPOINT)) {
            // enable the EP
            m_verb("enable the EP");
            m_ofi_call(fi_enable(trx[i].ep));
        }
        if (is_rx || is_sync) {
            // get the addresses from others
            m_verb("get the AV");
            m_rmem_call(ofi_util_av(comm->size, trx[i].ep, trx[i].av, &trx[i].addr));
        }
        //------------------------------------------------------------------------------------------
        // gpu specific
        m_gpu_call(gpuStreamCreate(&trx[i].stream));
        m_verb("done with EP # %d", i);
        m_verb("-----------------");
    }

    //---------------------------------------------------------------------------------------------
    // if needed, enable the MR and then get the corresponding key and share it
    // first the user region's key
    m_rmem_call(ofi_util_mr_enable(mem->ofi.mr.mr, comm, &mem->ofi.mr.key_list));
    if (comm->prov_mode.rtr_mode == M_OFI_RTR_ATOMIC) {
        ofi_util_sig_enable(&mem->ofi.sync.ps_sig, comm);
    }

    //---------------------------------------------------------------------------------------------
    // allocate the data user for sync
    // we need one ctx entry per rank
    // all of them refer to the same sync epoch
    mem->ofi.sync.cqdata_ps = calloc(comm->size, sizeof(ofi_cqdata_t));
    mem->ofi.sync.cqdata_cw = calloc(comm->size, sizeof(ofi_cqdata_t));
    ofi_cqdata_t* tmp_cq_array[2] = {mem->ofi.sync.cqdata_ps, mem->ofi.sync.cqdata_cw};
    for (int i = 0; i < comm->size; ++i) {
        for (int j = 0; j < 2; ++j) {
            ofi_cqdata_t* ccq = tmp_cq_array[j] + i;
            ccq->kind = m_ofi_cq_kind_sync;
            ccq->epoch_ptr = mem->ofi.sync.epch;
            m_verb("registering sync memory [%d,%d]", i, j);
            m_rmem_call(ofi_util_mr_reg(&ccq->sync.data, sizeof(uint64_t),
                                        FI_SEND | FI_RECV | FI_WRITE, comm, &ccq->sync.mr.mr,
                                        &ccq->sync.mr.desc, NULL));
            m_rmem_call(ofi_util_mr_bind(mem->ofi.sync_trx->ep, ccq->sync.mr.mr, NULL, comm));
            m_rmem_call(ofi_util_mr_enable(ccq->sync.mr.mr, comm, NULL));
        }
    }
    //----------------------------------------------------------------------------------------------
    // post the AM buffers once the EP are ready done
    if (comm->prov_mode.rtr_mode == M_OFI_RTR_MSG || comm->prov_mode.dtc_mode == M_OFI_DTC_MSG) {
        m_verb("init AM buffers");
        ofi_rmem_am_init(mem, comm);
        // we MUST have this barrier to make sure that no sync message is sent before all the AM
        // receive have been posted
        PMI_Barrier();
    }
    //---------------------------------------------------------------------------------------------
    // allocate trigr pool
    // m_countr_init(&mem->ofi.qtrigr.trigr_count);
    // // allocate the bitmap to track completion of the requests
    // mem->ofi.qtrigr.pool_bitmap = m_malloc(m_gpu_n_trigr / 8 + (m_gpu_n_trigr % 8) > 0);
    // if (M_HAVE_GPU) {
    //     m_gpu_call(gpuHostAlloc((void**)&mem->ofi.qtrigr.h_trigr_pool, m_gpu_page_size,
    //                             gpuHostAllocMapped));
    //     m_gpu_call(gpuHostGetDevicePointer((void**)&mem->ofi.qtrigr.d_trigr_pool,
    //                                        (void*)mem->ofi.qtrigr.h_trigr_pool, 0));
    // } else {
    //     mem->ofi.qtrigr.h_trigr_pool = m_malloc(m_gpu_page_size);
    //     mem->ofi.qtrigr.d_trigr_pool = mem->ofi.qtrigr.h_trigr_pool;
    // }
    // // async progress
    // m_countr_init(&mem->ofi.qtrigr.ongoing);
    // m_atomicptr_init(&mem->ofi.qtrigr.head);
    // m_atomicptr_init(&mem->ofi.qtrigr.tail);
    // m_atomicptr_init(&mem->ofi.qtrigr.prev);
    // m_atomicptr_init(&mem->ofi.qtrigr.curnt);
    rmem_lmpsc_create(&mem->ofi.qtrigr);
    // create the thread
    pthread_attr_t pthread_attr;
    m_pthread_call(pthread_attr_init(&pthread_attr));
    mem->ofi.thread_arg = (rmem_thread_arg_t){
        .workq = &mem->ofi.qtrigr,
        .data_trx = mem->ofi.data_trx,
        .xctx.epoch_ptr = mem->ofi.sync.epch,
        .n_tx = mem->ofi.n_tx,
    };

    // create and lock the progress on the thread
    mem->ofi.thread_arg.do_progress = m_malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mem->ofi.thread_arg.do_progress, NULL);
    pthread_mutex_lock(mem->ofi.thread_arg.do_progress);
    // m_countr_init(mem->ofi.thread_arg.do_progress);
    m_pthread_call(
        pthread_create(&mem->ofi.progress, &pthread_attr, &ofi_tthread_main, &mem->ofi.thread_arg));
    m_pthread_call(pthread_attr_destroy(&pthread_attr));
    //---------------------------------------------------------------------------------------------
    // GPU
#ifndef NDEBUG
    int device_count = 0;
    m_gpu_call(gpuGetDeviceCount(&device_count));
    if (device_count > 1) {
        m_log("WARNING: more than one GPU per rank (here %d) is not supported", device_count);
    }
    m_gpu_call(gpuSetDevice(0));
#endif

    //----------------------------------------------------------------------------------------------
    return m_success;
}
int ofi_rmem_free(ofi_rmem_t* mem, ofi_comm_t* comm) {
    //----------------------------------------------------------------------------------------------
    // cancel the thread first to avoid issues
    void* retval = NULL;
    m_pthread_call(pthread_cancel(mem->ofi.progress));
    m_pthread_call(pthread_join(mem->ofi.progress, &retval));
    rmem_lmpsc_destroy(&mem->ofi.qtrigr);
    pthread_mutex_destroy(mem->ofi.thread_arg.do_progress);
    free(mem->ofi.thread_arg.do_progress);
    // if (M_HAVE_GPU) {
    //     m_gpu_call(gpuFreeHost((void*)mem->ofi.qtrigr.h_trigr_pool));
    // } else {
    //     free((void*)mem->ofi.qtrigr.h_trigr_pool);
    // }
    //----------------------------------------------------------------------------------------------
    if (comm->prov_mode.rtr_mode == M_OFI_RTR_MSG || comm->prov_mode.dtc_mode == M_OFI_DTC_MSG) {
        ofi_rmem_am_free(mem, comm);
    }
    //----------------------------------------------------------------------------------------------
    // free the sync
    for (int i = 0; i < comm->size; ++i) {
        m_verb("closing sync memory [%d,%d]", i, 0);
        m_rmem_call(ofi_util_mr_close(mem->ofi.sync.cqdata_ps[i].sync.mr.mr));
        m_verb("closing sync memory [%d,%d]", i, 1);
        m_rmem_call(ofi_util_mr_close(mem->ofi.sync.cqdata_cw[i].sync.mr.mr));
    }
    free(mem->ofi.sync.cqdata_ps);
    free(mem->ofi.sync.cqdata_cw);
    // free the MRs
    m_verb("closing user memory");
    m_rmem_call(ofi_util_mr_close(mem->ofi.mr.mr));
    free(mem->ofi.mr.key_list);
    free(mem->ofi.mr.base_list);

    // close the signals
    if (comm->prov_mode.rtr_mode == M_OFI_RTR_ATOMIC) {
        ofi_util_sig_close(&mem->ofi.sync.ps_sig);
    }

    // free the Tx first, need to close them before closing the AV in the Rx
    const int n_trx = comm->n_ctx + 1;
    for (int i = 0; i < n_trx; ++i) {
        const bool is_rx = (i < mem->ofi.n_rx);
        const bool is_tx = (i < mem->ofi.n_tx);
        const bool is_sync = (i == comm->n_ctx);

        ofi_rma_trx_t* trx = (is_sync) ? (mem->ofi.sync_trx) : (mem->ofi.data_trx + i);
        struct fid_ep* nullsrx = NULL;
        struct fid_stx* nullstx = NULL;
        m_rmem_call(ofi_util_free_ep(comm->prov, &trx->ep, &nullstx, &nullsrx));
        m_ofi_call(fi_close(&trx->cq->fid));
        if (is_rx || is_sync) {
            m_ofi_call(fi_close(&trx->av->fid));
            free(trx->addr);
        }
        m_gpu_call(gpuStreamDestroy(trx->stream));
    }
    if (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_REMOTE_CNTR) {
        m_ofi_call(fi_close(&mem->ofi.rcntr->fid));
    }
    free(mem->ofi.data_trx);
    // free the sync
    free(mem->ofi.sync.icntr);
    return m_success;
}

typedef enum {
    RMA_OPT_PUT,
    RMA_OPT_RPUT,
} rma_opt_t;

static int ofi_rma_init(ofi_rma_t* rma, ofi_rmem_t* mem, const int ctx_id, ofi_comm_t* comm,
                        rma_opt_t op) {
    m_assert(ctx_id < comm->n_ctx, "ctx id = %d < the number of ctx = %d", ctx_id, comm->n_ctx);
    //----------------------------------------------------------------------------------------------
    // endpoint and address
    const int rx_id = m_get_rx(ctx_id, mem);
    rma->ofi.ep = mem->ofi.data_trx[ctx_id].ep;
    rma->ofi.addr = mem->ofi.data_trx[rx_id].addr[rma->peer];
    //----------------------------------------------------------------------------------------------
    // if needed, register the memory
    m_verb("registering memory origin");
    m_rmem_call(ofi_util_mr_reg(rma->buf, rma->count, FI_WRITE | FI_READ, comm, &rma->ofi.msg.mr.mr,
                                &rma->ofi.msg.mr.desc, NULL));
    m_rmem_call(ofi_util_mr_bind(rma->ofi.ep, rma->ofi.msg.mr.mr, NULL, comm));
    m_rmem_call(ofi_util_mr_enable(rma->ofi.msg.mr.mr, comm, NULL));

    //----------------------------------------------------------------------------------------------
    // IOVs
    rma->ofi.msg.iov = (struct iovec){
        .iov_base = rma->buf,
        .iov_len = rma->count,
    };
    m_verb("rma-init: base = %llu + disp = %lu", mem->ofi.mr.base_list[rma->peer], rma->disp);
    rma->ofi.msg.riov = (struct fi_rma_iov){
        .addr = mem->ofi.mr.base_list[rma->peer] + rma->disp,  // offset from key
        .len = rma->count,
        .key = mem->ofi.mr.key_list[rma->peer],
    };
    //----------------------------------------------------------------------------------------------
    // cq and progress
    // any of the cqdata entry can be used to fallback, the first one always exists
    rma->ofi.progress.cq = mem->ofi.data_trx[ctx_id].cq;
    rma->ofi.progress.xctx.epoch_ptr = mem->ofi.sync.epch;
    switch (op) {
        case (RMA_OPT_PUT): {
            m_verb("using kind local and null");
            rma->ofi.msg.cq.kind = m_ofi_cq_inc_local | m_ofi_cq_kind_null;
        } break;
        case (RMA_OPT_RPUT): {
            m_verb("using kind local and rqst");
            rma->ofi.msg.cq.kind = m_ofi_cq_inc_local | m_ofi_cq_kind_rqst;
        } break;
    }
    rma->ofi.msg.cq.epoch_ptr = mem->ofi.sync.epch;
    //----------------------------------------------------------------------------------------------
    // setup the ready flag to 1 to avoid issues
    rma->ofi.qnode.h_ready_ptr = 0;
    rma->ofi.qnode.kind = LNODE_KIND_RMA;
    //----------------------------------------------------------------------------------------------
    // flag
    const bool do_inject = false; //(rma->count < comm->prov->tx_attr->inject_size);
    const bool do_delivery = (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL);
    const bool auto_progress = (comm->prov->domain_attr->data_progress & FI_PROGRESS_AUTO);
    // force the use of delivery complete if needed
    m_verb("injection? %ld vs %ld", rma->count, comm->prov->tx_attr->inject_size);
    uint64_t flag_complete = (do_inject) ? FI_INJECT : 0x0;
    if (do_delivery) {
        flag_complete |= FI_DELIVERY_COMPLETE;
        m_assert(!do_inject, "we cannot inject at the same time");
        m_verb("using FI_DELIVERY_COMPLETE");
    } else if (auto_progress) {
        flag_complete |= FI_INJECT_COMPLETE;
        m_verb("using FI_INJECT_COMPLETE");
    } else {
        flag_complete |= FI_TRANSMIT_COMPLETE;
        m_assert(!do_inject, "we cannot inject at the same time");
        m_verb("using FI_TRANSMIT_COMPLETE");
    }
    // fill out the flags
    rma->ofi.msg.flags = (do_inject ? FI_INJECT : 0x0);
    switch (op) {
        case (RMA_OPT_PUT): {
            rma->ofi.msg.flags |= flag_complete;
        } break;
        case (RMA_OPT_RPUT): {
            rma->ofi.msg.flags |= flag_complete;
            rma->ofi.msg.flags |= FI_COMPLETION;
        } break;
    }
    // message data
    if (comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_CQ_DATA) {
        rma->ofi.msg.flags |= FI_REMOTE_CQ_DATA;
        rma->ofi.msg.data = m_ofi_data_set_rcq;
    } else {
        rma->ofi.msg.data = 0x0;
    }
    //---------------------------------------------------------------------------------------------
    // GPU request - allocated to the GPU
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy__zero-copy-host-code
    // volatile int* h_ready_ptr = &rma->ofi.qnode.h_ready_ptr;
    if (M_HAVE_GPU) {
        // int* d_ready_ptr;
        // m_gpu_call(
        //     gpuHostRegister((void*)h_ready_ptr, sizeof(volatile int), gpuHostRegisterMapped));
        // m_gpu_call(gpuHostGetDevicePointer((void**)&d_ready_ptr, (void*)h_ready_ptr, 0));
        // m_verb("RMA operation: host ptr is %p, device ptr is %p",h_ready_ptr,d_ready_ptr);

        // store the stream and the device pointer
        rma->ofi.stream = &mem->ofi.data_trx[ctx_id].stream;
        // rma->ofi.qnode.d_ready_ptr = d_ready_ptr;
    } else {
        rma->ofi.stream = NULL;
        // rma->ofi.qnode.d_ready_ptr = h_ready_ptr;
    }
    //----------------------------------------------------------------------------------------------
    m_assert(rma->ofi.msg.riov.key != FI_KEY_NOTAVAIL, "key must be >0");
    return m_success;
}
int ofi_put_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_PUT);
}
int ofi_rput_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_RPUT);
}
int ofi_rma_put_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_PUT);
}
int ofi_rma_rput_init(ofi_rma_t* put, ofi_rmem_t* pmem, const int ctx_id, ofi_comm_t* comm) {
    return ofi_rma_init(put, pmem, ctx_id, comm, RMA_OPT_RPUT);
}
int ofi_rma_enqueue(ofi_rmem_t* mem, ofi_rma_t* rma, rmem_trigr_ptr* trigr, rmem_device_t dev) {
    if (dev == RMEM_TRIGGER) {
        // enqueue the operation
        mem->ofi.sync.icntr[rma->peer]++;
        *trigr = rmem_lmpsc_enq(&mem->ofi.qtrigr, &rma->ofi.qnode);
        m_verb("enqueuing: trigger value = %p", trigr[0]);
    } else {
        rma->ofi.qnode.h_ready_ptr = NULL;
        rma->ofi.qnode.d_ready_ptr = NULL;
        trigr = NULL;
    }
    //----------------------------------------------------------------------------------------------
    return m_success;
}
int ofi_rma_reset_queue(ofi_rmem_t* mem){
    rmem_lmpsc_reset(&mem->ofi.qtrigr);
    return m_success;
}
int ofi_rma_start(ofi_rmem_t* mem, ofi_rma_t* rma, rmem_device_t dev) {
    m_assert(!(rma->ofi.qnode.h_ready_ptr && rma->ofi.qnode.h_ready_ptr[0] == 0),
             "the ready value must be 0 and not %lld", rma->ofi.qnode.h_ready_ptr[0]);
    if (dev == RMEM_TRIGGER) {
        m_verb("triggering %p",&rma->ofi.qnode);
        // trigger from the GPU, counters are incremented at enqueue time
        ofi_rma_start_gpu(rma->ofi.stream, rma->ofi.qnode.d_ready_ptr);
    } else {
        m_verb("starting %p",&rma->ofi.qnode);
        // trigger from the host: increment the counters first
        mem->ofi.sync.icntr[rma->peer]++;
        m_rmem_call(ofi_rma_start_from_task(&rma->ofi.qnode));
    }
    return m_success;
}

int ofi_rma_free(ofi_rma_t* rma) {
    m_verb("closing memory origin");
    m_rmem_call(ofi_util_mr_close(rma->ofi.msg.mr.mr));
    // m_gpu_call(gpuHostUnregister((void*)&rma->ofi.qnode.ready));
    return m_success;
}

#define m_ofi_rma_task_offset(a) (offsetof(ofi_rma_t, ofi.a) - offsetof(ofi_rma_t, ofi.qnode))
#define m_ofi_rma_task_structgetptr(T, name, a, task) \
    T* name = (T*)((uint8_t*)task + m_ofi_rma_task_offset(a));
int ofi_rma_start_from_task(rmem_lnode_t* task) {
    m_ofi_rma_task_structgetptr(struct fid_ep*, ep, ep, task);
    m_ofi_rma_task_structgetptr(fi_addr_t, addr, addr, task);
    //--------------------------------------------------------------------------------------
    // msg specific data
    m_ofi_rma_task_structgetptr(uint64_t, msg_flags, msg.flags, task);
    m_ofi_rma_task_structgetptr(uint64_t, msg_data, msg.data, task);
    m_ofi_rma_task_structgetptr(ofi_cqdata_t, msg_cq, msg.cq, task);
    m_ofi_rma_task_structgetptr(struct iovec, msg_iov, msg.iov, task);
    m_ofi_rma_task_structgetptr(struct fi_rma_iov, msg_riov, msg.riov, task);
    m_ofi_rma_task_structgetptr(ofi_progress_t, rma_prog, progress, task);
    m_ofi_rma_task_structgetptr(void*, msg_desc, msg.mr.desc, task);

    struct fi_msg_rma msg = {
        .msg_iov = msg_iov,
        .desc = msg_desc,  // it's a void** which is correct
        .iov_count = 1,
        .addr = *addr,
        .rma_iov = msg_riov,
        .rma_iov_count = 1,
        .data = *msg_data,
        .context = &msg_cq->ctx,
    };
    m_verb("THREAD: doing RMA on EP %p, size = %ld", *ep,msg_iov->iov_len);
    m_ofi_call_again(fi_writemsg(*ep, &msg, *msg_flags), rma_prog);
    // if we had to get a cq entry and the inject, mark is as done
    if ((*msg_flags) & FI_INJECT && (*msg_flags) & FI_COMPLETION) {
        m_countr_fetch_add(&msg_cq->rqst.busy, -1);
    }
    return m_success;
}
// end of file
