/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "rmem_run.h"

#include <math.h>

#include "pmi.h"
#include "rmem_profile.h"

#define n_measure       50
#define n_warmup        5
#define retry_threshold 0.05
#define retry_max       2
#define bar_max         50

//==================================================================================================
void run_test(run_t* sender, run_t* recver, run_param_t param, run_time_t* timings) {
    // retrieve useful parameters
    const int n_size = log10(param.msg_size) / log10(2.0) + 1;
    const int n_msg = log10(param.n_msg)/ log10(2.0) +1;
    size_t ttl_sample = n_msg * n_size;
    ofi_comm_t* comm = param.comm;
    // allocate the results
    timings->avg = malloc(sizeof(double) * ttl_sample);
    timings->ci = malloc(sizeof(double) * ttl_sample);
    // get the retry value
    int* retry_ptr = malloc(sizeof(int));
    ofi_p2p_t p2p_retry = {
        .peer = peer(comm->rank, comm->size),
        .buf = retry_ptr,
        .count = sizeof(int),
        .tag = param.n_msg + 1,
    };
    if(is_sender(comm->rank)){
        ofi_recv_init(&p2p_retry, 0, comm);
    } else {
        ofi_send_init(&p2p_retry, 0, comm);
    }
    //----------------------------------------------------------------------------------------------
    for (int imsg = 1; imsg <= param.n_msg; imsg *= 2) {
        const int idx_msg = log10(imsg) / log10(2.0);
        for (size_t msg_size = 1; msg_size <= param.msg_size; msg_size *= 2) {
            const int idx_size = log10(msg_size) / log10(2.0);
            if (imsg * msg_size * sizeof(int) > m_max_size) {
                break;
            }
            int idx = idx_msg * n_size + idx_size;
            run_param_t cparam = {
                .msg_size = msg_size,
                .comm = param.comm,
                .mem = param.mem,
                .n_msg = imsg,
            };
            m_verb("memory -----------------");
            PMI_Barrier();
            if (is_sender(comm->rank)) {
                //---------------------------------------------------------------------------------
                //- SENDER
                //---------------------------------------------------------------------------------
                sender->pre(&cparam, sender->data);
                // loop
                *retry_ptr = 1;
                while (*retry_ptr) {
                    for (int it = -n_warmup; it < n_measure; ++it) {
                        sender->run(&cparam, sender->data);
                    }
                    ofi_p2p_start(&p2p_retry);
                    ofi_p2p_wait(&p2p_retry);
                }
                sender->post(&cparam, sender->data);
            } else {
                //---------------------------------------------------------------------------------
                //- RECVER
                //---------------------------------------------------------------------------------
                recver->pre(&cparam, recver->data);
                // profile stuff
                *retry_ptr = 1;
                while (*retry_ptr) {
                    double time[n_measure];
                    for (int it = -n_warmup; it < n_measure; ++it) {
                        time[(it >= 0) ? it : 0] = recver->run(&cparam, recver->data);
                    }
                    // get the CI + the retry
                    double tavg, ci;
                    rmem_get_ci(n_measure, time, &tavg, &ci);
                    m_verb("msg = %ld B: avg = %f, CI = %f, ratio = %f vs %f retry = %d/%d",
                           msg_size, tavg, ci, ci / tavg, retry_threshold, *retry_ptr, retry_max);
                    // store the results
                    m_assert(idx < ttl_sample, "ohoh: id = %d, ttl_sample = %ld", idx, ttl_sample);
                    timings->avg[idx] = tavg;
                    timings->ci[idx] = ci;
                    // retry?
                    if (*retry_ptr > retry_max || (ci / tavg) < retry_threshold) {
                        *retry_ptr = 0;
                    } else {
                        (*retry_ptr)++;
                    }
                    // send the information to the sender side
                    ofi_p2p_start(&p2p_retry);
                    ofi_p2p_wait(&p2p_retry);
                }
                recver->post(&cparam, recver->data);
            }
        }
    }
    m_rmem_call(ofi_p2p_free(&p2p_retry));
    free(retry_ptr);
}
//==================================================================================================
//= POINT TO POINT
//==================================================================================================

static void p2p_pre_alloc(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    // allocat the buff
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    // allocate the buf
    d->buf = calloc(ttl_len, sizeof(int));
    for (int i = 0; i < ttl_len; ++i) {
        d->buf[i] = i + 1;
    }
    // allocate the objects
    d->p2p = calloc(param->n_msg, sizeof(ofi_p2p_t));
    for (int i = 0; i < param->n_msg; ++i) {
        d->p2p[i] = (ofi_p2p_t){
            .buf = d->buf + i * msg_size,
            .count = msg_size * sizeof(int),
            .peer = peer(param->comm->rank, param->comm->size),
            .tag = i,
        };
    }
}
void p2p_pre_send(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    p2p_pre_alloc(param,data);
    // allocat the buff
    for (int i = 0; i < param->n_msg; ++i) {
        ofi_send_init(d->p2p + i, 0, param->comm);
    }
}
void p2p_pre_recv(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    p2p_pre_alloc(param,data);
    // allocat the buff
    for (int i = 0; i < param->n_msg; ++i) {
        ofi_recv_init(d->p2p + i, 0, param->comm);
    }
}
static void p2p_dealloc(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    // allocat the buff
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    for (int i = 0; i < n_msg; ++i) {
        ofi_p2p_free(d->p2p + i);
    }
    free(d->p2p);
    free(d->buf);
}
void p2p_post_send(run_param_t* param, void* data) {
    p2p_dealloc(param,data);
}
void p2p_post_recv(run_param_t* param, void* data) {
    p2p_dealloc(param,data);
}
double p2p_run_send(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;

    PMI_Barrier();
    // start exposure
    for (int j = 0; j < n_msg; ++j) {
        ofi_p2p_start(p2p + j);
    }
    for (int j = 0; j < n_msg; ++j) {
        ofi_p2p_wait(p2p + j);
    }
    return 0.0;
}
double p2p_run_recv(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    PMI_Barrier();
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_start(p2p + j);
        }
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_wait(p2p + j);
        }
    }
    //------------------------------------------------
    // check the result
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(d->buf[i] == res, "pmem[%d] = %d != %d", i, d->buf[i], res);
#else
        if (d->buf[i] != res) {
            m_log("pmem[%d] = %d != %d", i, d->buf[i], res);
        }
#endif
        d->buf[i] = 0;
    }
    return time;
}
double p2p_fast_run_send(run_param_t* param, void* data) {
    p2p_run_send(param, data);
    return 0.0;
}
double p2p_fast_run_recv(run_param_t* param, void* data) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    for (int j = 0; j < n_msg; ++j) {
        ofi_p2p_start(p2p + j);
    }
    PMI_Barrier();
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_wait(p2p + j);
        }
    }
    //------------------------------------------------
    // check the result
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(d->buf[i] == res, "pmem[%d] = %d != %d", i, d->buf[i], res);
#else
        if (d->buf[i] != res) {
            m_log("pmem[%d] = %d != %d", i, d->buf[i], res);
        }
#endif
        d->buf[i] = 0;
    }
    return time;
}
//==================================================================================================
//= RMA
//==================================================================================================

void rma_alloc(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    if (is_sender(param->comm->rank)) {
        // sender needs the local buffer
        d->buf = calloc(ttl_len, sizeof(int));
        for (int i = 0; i < ttl_len; ++i) {
            d->buf[i] = i + 1;
        }
    }
}
void rma_dealloc(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    if (is_sender(param->comm->rank)) {
        free(d->buf);
    }
}

void put_pre_send(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    // get the allocation of buffers
    rma_alloc(param, data);
    // allocate the puts
    d->rma = calloc(n_msg, sizeof(ofi_rma_t));
    for (int i = 0; i < n_msg; ++i) {
        d->rma[i] = (ofi_rma_t){
            .buf = d->buf + i * msg_size,
            .count = msg_size * sizeof(int),
            .disp = i * msg_size * sizeof(int),
            .peer = peer(param->comm->rank, param->comm->size),
        };
        ofi_rma_put_init(d->rma + i, param->mem, 0, param->comm);
    }
}

void sig_pre_send(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    // get the allocation of buffers
    rma_alloc(param, data);
    // allocate the puts
    d->rma = calloc(n_msg, sizeof(ofi_rma_t));
    for (int i = 0; i < n_msg; ++i) {
        d->rma[i] = (ofi_rma_t){
            .buf = d->buf + i * msg_size,
            .count = msg_size * sizeof(int),
            .disp = i * msg_size * sizeof(int),
            .peer = peer(param->comm->rank, param->comm->size),
        };
        ofi_rma_put_signal_init(d->rma + i, param->mem, 0, param->comm);
    }
}

void rma_pre_recv(run_param_t* param, void* data) {
    // get the allocation of buffers
    rma_alloc(param, data);
}

//--------------------------------------------------------------------------------------------------
void rma_post(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;

    if (is_sender(param->comm->rank)) {
        const int n_msg = param->n_msg;
        for (int i = 0; i < n_msg; ++i) {
            ofi_rma_free(d->rma + i);
        }
        free(d->rma);
    }
    rma_dealloc(param, data);
}

//--------------------------------------------------------------------------------------------------
double rma_run_send(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    PMI_Barrier();  // start exposure
    ofi_rmem_start(1, &buddy, param->mem, param->comm);
    for (int j = 0; j < n_msg; ++j) {
        ofi_rma_start(param->mem, d->rma + j);
    }
    ofi_rmem_complete(1, &buddy, param->mem, param->comm);
    return 0.0;
}
double rma_fast_run_send(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    PMI_Barrier();  // start exposure
    ofi_rmem_start_fast(1, &buddy, param->mem, param->comm);
    for (int j = 0; j < n_msg; ++j) {
        ofi_rma_start(param->mem, d->rma + j);
    }
    ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
    return 0.0;
}
double lat_run_send(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    ofi_rmem_start_fast(1, &buddy, param->mem, param->comm);
    PMI_Barrier();  // start exposure
    for (int j = 0; j < n_msg; ++j) {
        ofi_rma_start(param->mem, d->rma + j);
    }
    ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
    return 0.0;
}

//--------------------------------------------------------------------------------------------------
double rma_run_recv(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    PMI_Barrier();
    m_rmem_prof(prof, time) {
        ofi_rmem_post(1, &buddy, param->mem, param->comm);
        ofi_rmem_wait(1, &buddy, param->mem, param->comm);
    }
    //------------------------------------------------
    // check the result
    int* mem_buf = (int*)param->mem->buf;
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(mem_buf[i] == res, "pmem[%d] = %d != %d", i, mem_buf[i], res);
#else
        if (mem_buf[i] != res) {
            m_log("pmem[%d] = %d != %d", i, mem_buf[i], res);
        }
#endif
        mem_buf[i] = 0;
    }
    return time;
}
double rma_fast_run_recv(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    PMI_Barrier();
    m_rmem_prof(prof, time) {
        ofi_rmem_post_fast(1, &buddy, param->mem, param->comm);
        ofi_rmem_wait_fast(n_msg, param->mem, param->comm);
    }
    //------------------------------------------------
    // check the result
    int* mem_buf = (int*)param->mem->buf;
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(mem_buf[i] == res, "pmem[%d] = %d != %d", i, mem_buf[i], res);
#else
        if (mem_buf[i] != res) {
            m_log("pmem[%d] = %d != %d", i, mem_buf[i], res);
        }
#endif
        mem_buf[i] = 0;
    }
    return time;
}

double sig_run_recv(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    PMI_Barrier();
    m_rmem_prof(prof, time) {
        ofi_rmem_post(1, &buddy, param->mem, param->comm);
        ofi_rmem_sig_wait(n_msg, param->mem, param->comm);
        ofi_rmem_wait(1, &buddy, param->mem, param->comm);
    }
    //------------------------------------------------
    // check the result
    int* mem_buf = (int*)param->mem->buf;
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(mem_buf[i] == res, "pmem[%d] = %d != %d", i, mem_buf[i], res);
#else
        if (mem_buf[i] != res) {
            m_log("pmem[%d] = %d != %d", i, mem_buf[i], res);
        }
#endif
        mem_buf[i] = 0;
    }
    return time;
}
double lat_run_recv(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    ofi_rmem_post_fast(1, &buddy, param->mem, param->comm);
    PMI_Barrier();
    m_rmem_prof(prof, time) { ofi_rmem_wait_fast(n_msg, param->mem, param->comm); }
    //------------------------------------------------
    // check the result
    int* mem_buf = (int*)param->mem->buf;
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(mem_buf[i] == res, "pmem[%d] = %d != %d", i, mem_buf[i], res);
#else
        if (mem_buf[i] != res) {
            m_log("pmem[%d] = %d != %d", i, mem_buf[i], res);
        }
#endif
        mem_buf[i] = 0;
    }
    return time;
}
