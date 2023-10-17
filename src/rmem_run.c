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
#define retry_threshold 0.025
#define retry_max       10
#define n_repeat_offset 10

#define m_min_msg        1
#define m_min_size       1
#define m_msg_idx(imsg)  ((log10(imsg) - log10(m_min_msg)) / log10(2))
#define m_size_idx(imsg) ((log10(imsg) - log10(m_min_size)) / log10(2))

static void run_test_check(const size_t ttl_len, int* buf) {
    //------------------------------------------------
    // check the result
    int* tmp;
    if (M_HAVE_GPU) {
        tmp = calloc(ttl_len, sizeof(int));
        m_gpu_call(gpuMemcpySync(tmp, buf, ttl_len * sizeof(int), gpuMemcpyDeviceToHost));
    } else {
        tmp = buf;
    }
    for (int i = 0; i < ttl_len; ++i) {
        int res = i + 1;
#ifndef NDEBUG
        m_assert(tmp[i] == res, "pmem[%d] = %d != %d", i, tmp[i], res);
#else
        if (tmp[i] != res) {
            m_log("pmem[%d] = %d != %d", i, tmp[i], res);
        }
#endif
        tmp[i] = 0;
    }
    if (M_HAVE_GPU) {
        // copies all the zeros back
        m_gpu_call(gpuMemcpySync(buf, tmp, ttl_len * sizeof(int), gpuMemcpyHostToDevice));
        free(tmp);
    }
}
//==================================================================================================
typedef struct {
    union {
        double val;
        struct timespec time;
    } tmp;
    ofi_p2p_t send;
    ofi_p2p_t recv;
} ack_t;
static void ack_init(ack_t* ack, ofi_comm_t* comm) {
    ack->tmp.val = 0;
    ack->send = (ofi_p2p_t){
        .buf = &ack->tmp,
        .count = sizeof(ack->tmp),
        .tag = 0xffffffff,
        .peer = (comm->rank + 1) % comm->size,
    };
    ack->recv = (ofi_p2p_t)ack->send;
    ofi_send_init(&ack->send, 0, comm);
    ofi_recv_init(&ack->recv, 0, comm);
}
static void ack_send(ack_t* ack) {
    ofi_p2p_start(&ack->send);
    ofi_p2p_wait(&ack->send);
}
static void ack_wait(ack_t* ack) {
    ofi_p2p_start(&ack->recv);
    ofi_p2p_wait(&ack->recv);
}
static void ack_send_withval(ack_t* ack, double val) {
    ack->tmp.val = val;
    ofi_p2p_start(&ack->send);
    ofi_p2p_wait(&ack->send);
}
static double ack_wait_withval(ack_t* ack) {
    ofi_p2p_start(&ack->recv);
    ofi_p2p_wait(&ack->recv);
    return ack->tmp.val;
}
static void ack_send_withtime(ack_t* ack, struct timespec* t) {
    ack->tmp.time.tv_sec = t->tv_sec;
    ack->tmp.time.tv_nsec = t->tv_nsec;
    ofi_p2p_start(&ack->send);
    ofi_p2p_wait(&ack->send);
}
static void ack_wait_withtime(ack_t* ack, struct timespec* t) {
    ofi_p2p_start(&ack->recv);
    ofi_p2p_wait(&ack->recv);
    t->tv_sec = ack->tmp.time.tv_sec;
    t->tv_nsec = ack->tmp.time.tv_nsec;
}
static void ack_free(ack_t* ack) {
    ofi_p2p_free(&ack->send);
    ofi_p2p_free(&ack->recv);
}

static void ack_offset_sender(ack_t* ack) {
    // start with exposure epoch with the reference time
    ack_send(ack);
    for (int i = 0; i < n_repeat_offset; ++i) {
        // wait to recv t1
        ack_wait(ack);
        // obtain t2 and send it
        struct timespec t2;
        m_gettime(&t2);
        ack_send_withtime(ack, &t2);
    }
    // wait for recv side completion
    ack_wait(ack);
}
/**
 * @brief compute the offset in clocks for the receiver
 *
 * offset is defined as T_sender = T_receiver + o
 */
static double ack_offset_recver(ack_t* ack) {
    struct timespec t1, t2, t3;
    struct timespec tseed;
    double offset = 0.0;
    m_gettime(&tseed);
    // wait for the sender side to be ready
    ack_wait(ack);
    for (int i = 0; i < n_repeat_offset; ++i) {
        m_gettime(&t1);
        ack_send(ack);
        // receive T2 and get t3
        ack_wait_withtime(ack, &t2);
        m_gettime(&t3);
        // d = latency, o = offset: T_sender = T_receiver + o
        // T2 - (T1 + o) = d
        // (T3 + o) - T2 = d
        // (T3 + o - T2) - (T2 - T1 - o) = o
        // <=> T3 + 2 o - 2 T2 + T1 = 0
        // <=> T3 + 2 o - 2 T2 + T1 = 0
        // => o = T2 - (T3+T1)/2
        // note: we take the diff with the seed first to avoid overflow
        double wt1 = m_get_wtimes(tseed, t1);
        double wt2 = m_get_wtimes(tseed, t2);
        double wt3 = m_get_wtimes(tseed, t3);
        offset += (wt2 - 0.5 * (wt1 + wt3)) / n_repeat_offset;
    }
    // notify completion
    ack_send(ack);
    m_verb("average offset = %f", offset);
    return offset;
}
//==================================================================================================
void run_test(run_t* sender, run_t* recver, run_param_t param, run_time_t* timings) {
    // retrieve useful parameters
    const int n_msg = m_msg_idx(param.n_msg) + 1;
    const int n_size = m_size_idx(param.msg_size) + 1;
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
    ack_t ack;
    ack_init(&ack, param.comm);
    //----------------------------------------------------------------------------------------------
    for (int imsg = m_min_msg; imsg <= param.n_msg; imsg *= 2) {
        if (is_sender(comm->rank)) {
            m_log("-> doing now %d msgs ",imsg);
        }
        const int idx_msg = m_msg_idx(imsg);
        const size_t max_msg_size = m_msg_size(imsg, param.msg_size, int);
        for (size_t msg_size = m_min_size; msg_size <= max_msg_size; msg_size *= 2) {
            const int idx_size = m_msg_idx(msg_size) ;
            // m_log("idx_size = %d, msg_size = %ld, max size = %ld", idx_size, msg_size,
            //       max_msg_size);
            int idx = idx_msg * n_size + idx_size;
            run_param_t cparam = {
                .msg_size = msg_size,
                .comm = param.comm,
                .mem = param.mem,
                .n_msg = imsg,
            };
            m_verb("memory %lu -----------------",msg_size);
            PMI_Barrier();
            double time[n_measure];
            if (is_sender(comm->rank)) {
                //---------------------------------------------------------------------------------
                //- SENDER
                //---------------------------------------------------------------------------------
                sender->pre(&cparam, sender->data);
                // loop
                *retry_ptr = 1;
                while (*retry_ptr) {
                    for (int it = -n_warmup; it < n_measure; ++it) {
                        time[(it >= 0) ? it : 0] = sender->run(&cparam, sender->data, &ack);
                    }
                    // post process time
                    double tavg, ci;
                    rmem_get_ci(n_measure, time, &tavg, &ci);
                    m_assert(idx < ttl_sample,
                             "ohoh: id = %d = %d * %d + %d, ttl_sample = %ld = %d * %d", idx,
                             idx_msg, n_size, idx_size, ttl_sample, n_msg, n_size);
                    timings->avg[idx] = tavg;
                    timings->ci[idx] = ci;
                    // get if we retry
                    ofi_p2p_start(&p2p_retry);
                    ofi_p2p_wait(&p2p_retry);
                    m_verb("retry? %d", *retry_ptr);
                }
                sender->post(&cparam, sender->data);
            } else {
                //---------------------------------------------------------------------------------
                //- RECVER
                //---------------------------------------------------------------------------------
                m_verb("receiver->pre: starting");
                recver->pre(&cparam, recver->data);
                m_verb("receiver->pre: done");
                // profile stuff
                *retry_ptr = 1;
                while (*retry_ptr) {
                    for (int it = -n_warmup; it < n_measure; ++it) {
                        time[(it >= 0) ? it : 0] = recver->run(&cparam, recver->data, &ack);
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
                    m_verb("retry? %d", *retry_ptr);
                    // send the information to the sender side
                    ofi_p2p_start(&p2p_retry);
                    ofi_p2p_wait(&p2p_retry);
                }
                recver->post(&cparam, recver->data);
            }
        }
    }
    ack_free(&ack);
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
    int* tmp= calloc(ttl_len, sizeof(int));
    for (int i = 0; i < ttl_len; ++i) {
        tmp[i] = i + 1;
    }

    if (M_HAVE_GPU) {
        m_gpu_call(gpuMalloc((void**)&d->buf, ttl_len * sizeof(int)));
        m_gpu_call(gpuMemcpySync(d->buf, tmp, ttl_len * sizeof(int), gpuMemcpyHostToDevice));
        free(tmp);
    } else {
        d->buf = tmp;
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
    if (M_HAVE_GPU) {
        m_gpu_call(gpuFree(d->buf));
    } else {
        free(d->buf);
    }
}
void p2p_post_send(run_param_t* param, void* data) {
    p2p_dealloc(param,data);
}
void p2p_post_recv(run_param_t* param, void* data) {
    p2p_dealloc(param,data);
}
double p2p_run_send(run_param_t* param, void* data, void* ack_ptr) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;

    double time;
    rmem_prof_t prof = {.name = "send"};

    //------------------------------------------------
    ack_offset_sender(ack);
    // start exposure
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_start(p2p + j);
        }
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_wait(p2p + j);
        }
    }
    // send the starting time from the profiler
    ack_send_withtime(ack, &prof.t0);
    return time;
}
double p2p_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    const double offset = ack_offset_recver(ack);
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_start(p2p + j);
        }
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_wait(p2p + j);
        }
    }
    // T sender = t recver + offset => T recver = T sender - offset
    // time elapsed = (T recver -  (T sender - offset))
    struct timespec tsend;
    ack_wait_withtime(ack, &tsend);
    double sync_time = m_get_wtimes(tsend, prof.t1) + offset;
    m_verb("estimated time of comms = %f vs previously measured one %f, offset is %f", sync_time,
           time, offset);

    //------------------------------------------------
    // check the result
    run_test_check(ttl_len, d->buf);
    return sync_time;
}
double p2p_fast_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_send(param, data, ack_ptr);
}
double p2p_fast_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
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
    const double offset = ack_offset_recver(ack);
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_wait(p2p + j);
        }
    }
    // T sender = t recver + offset => T recver = T sender - offset
    // time elapsed = (T recver -  (T sender - offset))
    struct timespec tsend;
    ack_wait_withtime(ack, &tsend);
    double sync_time = m_get_wtimes(tsend, prof.t1) + offset;
    m_verb("estimated time of comms = %f vs previously measured one %f, offset is %f", sync_time,
           time, offset);

    //------------------------------------------------
    // check the result
    run_test_check(ttl_len, d->buf);
    return sync_time;
}
//==================================================================================================
//= RMA
//==================================================================================================
void rma_alloc(run_param_t* param, void* data) {
    m_verb("entering rma_alloc");
    run_rma_data_t* d = (run_rma_data_t*)data;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    if (is_sender(param->comm->rank)) {
        // sender needs the local buffer
        int* tmp = calloc(ttl_len, sizeof(int));
        for (int i = 0; i < ttl_len; ++i) {
            tmp[i] = i + 1;
        }
        if (M_HAVE_GPU) {
            m_gpu_call(gpuMalloc((void**)&d->buf, ttl_len * sizeof(int)));
            m_gpu_call(gpuMemcpySync(d->buf, tmp, ttl_len * sizeof(int), gpuMemcpyHostToDevice));
            free(tmp);
        } else {
            d->buf = tmp;
        }
    }
}
void rma_dealloc(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    if (is_sender(param->comm->rank)) {
        if (M_HAVE_GPU) {
            m_gpu_call(gpuFree(d->buf));
        } else {
            free(d->buf);
        }
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

void put_pre_recv(run_param_t* param, void* data) {
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
double rma_run_send_device(run_param_t* param, void* data, void* ack_ptr, rmem_device_t device) {
    m_verb("rma_run_send_device: entering");
    run_rma_data_t* d = (run_rma_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    double time;
    rmem_prof_t prof = {.name = "send"};
    //-------------------------------------------------
    // enqueue the requests
    for (int j = 0; j < n_msg; ++j) {
        ofi_rma_enqueue(param->mem, d->rma + j,device);
    }
    // send a readiness signal
    ack_send(ack);
    // start the request
    ofi_rmem_start(1, &buddy, param->mem, param->comm);
    // injection can only be measure on the time to put the msgs and complete
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_rma_start(param->mem, d->rma + j, device);
        }
        m_verb("rma_run_send_device: rmem_complete");
        ofi_rmem_complete(1, &buddy, param->mem, param->comm);
    }
    return time;
}
double rma_run_send_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_send_device(param, data, ack_ptr, RMEM_GPU);
}
double rma_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_send_device(param, data, ack_ptr, RMEM_HOST);
}
double rma_fast_run_send_device(run_param_t* param, void* data,void* ack_ptr,rmem_device_t device) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    ack_t* ack = (ack_t*) ack_ptr;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    // we cannot do the fast completion with FENCE or DELIVERY COMPLETE
    if (param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL ||
        param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE) {
        return 0.0;
    }

    double time;
    rmem_prof_t prof = {.name = "send"};
    //------------------------------------------------
    // enqueue the requests
    for (int j = 0; j < n_msg; ++j) {
        ofi_rma_enqueue(param->mem, d->rma + j,device);
    }
    // send a readiness signal
    ofi_rmem_start_fast(1, &buddy, param->mem, param->comm);
    // compute the time difference
    ack_offset_sender(ack);
    m_rmem_prof(prof, time) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_rma_start(param->mem, d->rma + j, device);
        }
        ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
    }
    // send the starting time from the profiler
    ack_send_withtime(ack, &prof.t0);
    return time;
}
double rma_fast_run_send_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_fast_run_send_device(param, data, ack_ptr, RMEM_GPU);
}
double rma_fast_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return rma_fast_run_send_device(param, data, ack_ptr, RMEM_HOST);
}
// double lat_run_send(run_param_t* param, void* data,void* ack_ptr) {
//     run_rma_data_t* d = (run_rma_data_t*)data;
//     ack_t* ack = (ack_t*) ack_ptr;
//     const int n_msg = param->n_msg;
//     const size_t msg_size = param->msg_size;
//     const size_t ttl_len = n_msg * msg_size;
//     const int buddy = peer(param->comm->rank, param->comm->size);
//     double time;
//     rmem_prof_t prof = {.name = "send"};
//     //------------------------------------------------
//     ofi_rmem_start_fast(1, &buddy, param->mem, param->comm);
//     ack_send(ack);
//     m_rmem_prof(prof, time) {
//         for (int j = 0; j < n_msg; ++j) {
//             ofi_rma_start(param->mem, d->rma + j);
//         }
//         ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
//     }
//     return time;
// }

//--------------------------------------------------------------------------------------------------
double rma_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    // wait for the readiness signal
    ack_wait(ack);
    m_rmem_prof(prof, time) {
        ofi_rmem_post(1, &buddy, param->mem, param->comm);
        ofi_rmem_wait(1, &buddy, param->mem, param->comm);
    }
    //------------------------------------------------
    // check the result
    run_test_check(ttl_len, param->mem->buf);
    return time;
}
double rma_run_recv_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_recv(param, data, ack_ptr);
}
double rma_fast_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    // we cannot do the fast completion with FENCE or DELIVERY COMPLETE
    if (param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL ||
        param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE) {
        return 0.0;
    }

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    ofi_rmem_post_fast(1, &buddy, param->mem, param->comm);
    // obtain the acknowledgment
    const double offset = ack_offset_recver(ack);
    m_rmem_prof(prof, time) {
        ofi_rmem_wait_fast(n_msg, param->mem, param->comm);
    }
    // T sender = t recver + offset => T recver = T sender - offset
    // time elapsed = (T recver -  (T sender - offset))
    struct timespec tsend;
    ack_wait_withtime(ack, &tsend);
    double sync_time = m_get_wtimes(tsend, prof.t1) + offset;
    m_verb("estimated time of comms = %f vs previously measured one %f, offset is %f", sync_time,
           time, offset);

    //------------------------------------------------
    // check the result
    run_test_check(ttl_len, param->mem->buf);
    return sync_time;
}
double rma_fast_run_recv_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_fast_run_recv(param, data, ack_ptr);
}

