/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
#include "rmem_run.h"
#include "rmem_run_gpu.h"

#include <math.h>
#include <float.h>

#include "pmi.h"
#include "rmem_profile.h"

#define n_measure       50
#define n_warmup        5
#define retry_threshold 0.025
#define retry_max       1
#define n_repeat_offset 10

/**
 * @brief returns a random number [0; max[
 */
static int rmem_get_rand(const int max) {
    // get a random number between [0;RAND_MAX[.
    // note: the modulo is required because rand() returns [0;RAND_MAX]
    const int i = rand() % RAND_MAX;
    double r = (double)i / (double)RAND_MAX;
    return (int)(r * (double)(max));
}

static void run_test_check(const size_t ttl_len, int* buf) {
    //------------------------------------------------
    // check the result
    volatile int* tmp;
    if (M_HAVE_GPU) {
        tmp = calloc(ttl_len, sizeof(int));
        m_gpu_call(gpuMemcpySync((int*)tmp, buf, ttl_len * sizeof(int), gpuMemcpyDeviceToHost));
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
        m_gpu_call(gpuMemcpySync(buf, (int*)tmp, ttl_len * sizeof(int), gpuMemcpyHostToDevice));
        free((void*)tmp);
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
    m_assert(n_msg >= 0 && n_size >= 0, "the number of msgs = %d and size = %d must be >=0", n_msg,
             n_size);
    size_t ttl_sample = n_msg * n_size;

    ofi_comm_t* comm = param.comm;
    // allocate the results
    timings->avg = m_malloc(sizeof(double) * ttl_sample);
    timings->ci = m_malloc(sizeof(double) * ttl_sample);
    // get the retry value
    int* retry_ptr = m_malloc(2 * sizeof(int));
    ofi_p2p_t p2p_retry = {
        .peer = peer(comm->rank, comm->size),
        .buf = retry_ptr,
        .count = 2 * sizeof(int),
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
            const int idx_size = m_size_idx(msg_size) ;
            // m_verb("idx_size = %d, msg_size = %ld, max size = %ld", idx_size, msg_size,
            //       max_msg_size);
            int idx = idx_msg * n_size + idx_size;
            m_verb("idx = %d, idx_msg = %d, n_size = %d, idx_size = %d",idx,idx_msg,n_size,idx_size);
            run_param_t cparam = {
                .msg_size = msg_size,
                .comm = param.comm,
                .mem = param.mem,
                .n_msg = imsg,
            };
            m_verb("memory %lu -----------------",msg_size);
            PMI_Barrier();
            double time[n_measure];
            timings->avg[idx] = 0;
            timings->ci[idx] = DBL_MAX;
            if (is_sender(comm->rank)) {
                //---------------------------------------------------------------------------------
                //- SENDER
                //---------------------------------------------------------------------------------
                sender->pre(&cparam, sender->data);
                // loop
                retry_ptr[0] = 1;
                while (retry_ptr[0]) {
                    for (int it = -n_warmup; it < n_measure; ++it) {
                        time[(it >= 0) ? it : 0] = sender->run(&cparam, sender->data, &ack);
                    }
                    // post process time
                    double tavg, ci;
                    rmem_get_ci(n_measure, time, &tavg, &ci);
                    m_assert(idx < ttl_sample,
                             "ohoh: id = %d = %d * %d + %d, ttl_sample = %ld = %d * %d", idx,
                             idx_msg, n_size, idx_size, ttl_sample, n_msg, n_size);
                    // get if we retry
                    ofi_p2p_start(&p2p_retry);
                    ofi_p2p_wait(&p2p_retry);
                    // decide if we keep the measure or not
                    if (retry_ptr[1]) {
                        timings->avg[idx] = tavg;
                        timings->ci[idx] = ci;
                    }
                    m_verb("retry? %d, keep? %d", retry_ptr[0], retry_ptr[1]);
                }
                sender->post(&cparam, sender->data);
                m_verb("\t%ld B in %.1f usec", msg_size * sizeof(int), timings->avg[idx]);
            } else {
                //---------------------------------------------------------------------------------
                //- RECVER
                //---------------------------------------------------------------------------------
                m_verb("receiver->pre: starting");
                recver->pre(&cparam, recver->data);
                m_verb("receiver->pre: done");
                // profile stuff
                retry_ptr[0] = 1;
                while (retry_ptr[0]) {
                    for (int it = -n_warmup; it < n_measure; ++it) {
                        time[(it >= 0) ? it : 0] = recver->run(&cparam, recver->data, &ack);
                    }
                    // get the CI + the retry
                    double tavg, ci;
                    rmem_get_ci(n_measure, time, &tavg, &ci);
                    m_verb("msg = %ld B: avg = %f, CI = %f, ratio = %f vs %f retry = %d/%d",
                           msg_size, tavg, ci, ci / tavg, retry_threshold, retry_ptr[0], retry_max);
                    m_assert(idx < ttl_sample, "ohoh: id = %d, ttl_sample = %ld", idx, ttl_sample);
                    // keep the results?
                    if (timings->ci[idx] > ci) {
                        timings->avg[idx] = tavg;
                        timings->ci[idx] = ci;
                        retry_ptr[1] = 1;
                    } else {
                        retry_ptr[1] = 0;
                    }
                    // retry?
                    if (retry_ptr[0] > retry_max || (ci / tavg) < retry_threshold) {
                        retry_ptr[0] = 0;  // do not retry
                    } else {
                        retry_ptr[0]++;
                    }
                    m_verb("retry? %d keep? %d (ci = %f)", retry_ptr[0],retry_ptr[1],ci);
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

    m_gpu_call(gpuStreamCreate(&d->stream));
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
    m_gpu_call(gpuStreamDestroy(d->stream));
}
void p2p_post_send(run_param_t* param, void* data) {
    p2p_dealloc(param,data);
}
void p2p_post_recv(run_param_t* param, void* data) {
    p2p_dealloc(param,data);
}
static double p2p_run_send_common(run_param_t* param, void* data, void* ack_ptr, rmem_device_t dev,
                                  rmem_protocol_t protocol) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;

    double time;
    rmem_prof_t prof = {.name = "send"};

    // get a random starting point
    const int start_id = rmem_get_rand(n_msg);
    //------------------------------------------------
    ack_offset_sender(ack);
    // start exposure
    if (dev == RMEM_AWARE) {
        m_rmem_prof(prof, time) {
            for (int j = 0; j < n_msg; ++j) {
                const int id = (start_id + j) % n_msg;
                ofi_p2p_start(p2p + id);
            }
            for (int j = 0; j < n_msg; ++j) {
                ofi_p2p_wait(p2p + j);
            }
        }
    } else {
        m_rmem_prof(prof, time) {
            gpu_trigger_op(RMEM_GPU_P2P, start_id, n_msg, d->buf, param->msg_size, NULL, d->stream);
            m_gpu_call(gpuStreamSynchronize(d->stream));
            for (int j = 0; j < n_msg; ++j) {
                const int id = (start_id + j) % n_msg;
                ofi_p2p_start(p2p + id);
            }
            for (int j = 0; j < n_msg; ++j) {
                ofi_p2p_wait(p2p + j);
            }
        }
    }
    // send the starting time from the profiler
    ack_send_withtime(ack, &prof.t0);
    return time;
}
static double p2p_run_recv_common(run_param_t* param, void* data, void* ack_ptr, rmem_device_t dev,
                                  rmem_protocol_t protocol) {
    run_p2p_data_t* d = (run_p2p_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    ofi_p2p_t* p2p = d->p2p;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;

    double time;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    if (protocol == RMEM_FAST) {
        for (int j = 0; j < n_msg; ++j) {
            ofi_p2p_start(p2p + j);
        }
    }
    const double offset = ack_offset_recver(ack);
    m_rmem_prof(prof, time) {
        if (protocol == RMEM_DEFAULT) {
            for (int j = 0; j < n_msg; ++j) {
                ofi_p2p_start(p2p + j);
            }
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
double p2p_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_send_common(param, data, ack_ptr, RMEM_AWARE, RMEM_DEFAULT);
}
double p2p_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_recv_common(param, data, ack_ptr, RMEM_AWARE, RMEM_DEFAULT);
}
double p2p_run_send_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_send_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_DEFAULT);
}
double p2p_run_recv_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_recv_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_DEFAULT);
}
//============ p2p fast
double p2p_fast_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_send_common(param, data, ack_ptr, RMEM_AWARE, RMEM_FAST);
}
double p2p_fast_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_recv_common(param, data, ack_ptr, RMEM_AWARE, RMEM_FAST);
}
double p2p_fast_run_send_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_send_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_FAST);
}
double p2p_fast_run_recv_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return p2p_run_recv_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_FAST);
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
        m_gpu_call(gpuStreamCreate(&d->stream));
        if (M_HAVE_GPU) {
            // allocate the triggers
            m_gpu_call(gpuMalloc((void**)&d->trigr, n_msg * sizeof(rmem_trigr_ptr)));
            // allocate buffer on device + copy info into it
            m_gpu_call(gpuMalloc((void**)&d->buf, ttl_len * sizeof(int)));
            m_gpu_call(gpuMemcpySync(d->buf, tmp, ttl_len * sizeof(int), gpuMemcpyHostToDevice));
            free(tmp);
        } else {
            d->buf = tmp;
            d->trigr = m_malloc(sizeof(rmem_trigr_ptr) * n_msg);
        }
    }
}
void rma_dealloc(run_param_t* param, void* data) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    if (is_sender(param->comm->rank)) {
        if (M_HAVE_GPU) {
            m_gpu_call(gpuFree(d->buf));
            m_gpu_call(gpuFree(d->trigr));
        } else {
            free(d->buf);
            free((void*)d->trigr);
        }
        m_gpu_call(gpuStreamDestroy(d->stream));
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
static double rma_run_send_common(run_param_t* param, void* data, void* ack_ptr,
                                  rmem_device_t device, rmem_protocol_t protocol) {
    m_verb("rma_run_send_device: entering");
    run_rma_data_t* d = (run_rma_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    // we cannot do the fast completion with FENCE or DELIVERY COMPLETE
    bool do_real_fast = !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL) &&
                        !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_ORDER) &&
                        !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
    double time;
    rmem_prof_t prof = {.name = "send"};
    //-------------------------------------------------
    // enqueue the requests, need to store locally the gpu addresses to trigger and then copy them
    // back to the device
    rmem_trigr_ptr* trigr = d->trigr;
    if (gpuMemoryType((void*)d->trigr) != gpuMemoryTypeSystem) {
        trigr = m_malloc(sizeof(rmem_trigr_ptr) * n_msg);
    }
    for (int j = 0; j < n_msg; ++j) {
        ofi_rma_enqueue(param->mem, d->rma + j, trigr + j, device);
    }
    if (gpuMemoryType((void*)d->trigr) != gpuMemoryTypeSystem) {
        m_gpu_call(gpuMemcpySync((void*)d->trigr, trigr, n_msg * sizeof(rmem_trigr_ptr),
                                 gpuMemcpyHostToDevice));
        free((void*)trigr);
    }
    // get the start id
    const int start_id = rmem_get_rand(n_msg);

    if (protocol == RMEM_FAST) {
        // if we use fast, first do the start, then sync with the receiver
        if (do_real_fast) {
            ofi_rmem_start_fast(1, &buddy, param->mem, param->comm);
        } else {
            ofi_rmem_start(1, &buddy, param->mem, param->comm);
        }
        // the time is measured similarly to the p2p using the offset
        ack_offset_sender(ack);
    } else if (protocol == RMEM_DEFAULT) {
        // send a readiness signal triggers the time measurement on the recv
        ack_send(ack);
        // start the request
        ofi_rmem_start(1, &buddy, param->mem, param->comm);
    }
    if (device == RMEM_AWARE) {
        // only measure injection on the send side
        m_rmem_prof(prof, time) {
            for (int j = 0; j < n_msg; ++j) {
                const int id = (start_id + j) % n_msg;
                ofi_rma_start(param->mem, d->rma + id, RMEM_AWARE);
            }
            m_verb("rma_run_send_device: rmem_complete");
            if (protocol == RMEM_FAST && do_real_fast) {
                ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
            } else {
                ofi_rmem_complete(1, &buddy, param->mem, param->comm);
            }
        }
    } else {
        m_rmem_prof(prof, time) {
            gpu_trigger_op(RMEM_GPU_PUT, start_id, n_msg, d->buf, param->msg_size, d->trigr,
                           d->stream);
            m_verb("rma_run_send_device: rmem_complete");
            if (protocol == RMEM_FAST && do_real_fast) {
                ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
            } else {
                ofi_rmem_complete(1, &buddy, param->mem, param->comm);
            }
            m_gpu_call(gpuStreamSynchronize(d->stream));
        }
    }
    if (protocol == RMEM_FAST) {
        // send the starting time from the profiler
        ack_send_withtime(ack, &prof.t0);
    }
    ofi_rma_reset_queue(param->mem);
    return time;
}
double rma_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_send_common(param, data, ack_ptr, RMEM_AWARE, RMEM_DEFAULT);
}
double rma_run_send_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_send_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_DEFAULT);
}
double rma_fast_run_send(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_send_common(param, data, ack_ptr, RMEM_AWARE, RMEM_FAST);
}
double rma_fast_run_send_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_send_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_FAST);
}
// double rma_fast_run_send_device(run_param_t* param, void* data,void* ack_ptr,rmem_device_t device) {
//     run_rma_data_t* d = (run_rma_data_t*)data;
//     ack_t* ack = (ack_t*) ack_ptr;
//     const int n_msg = param->n_msg;
//     const size_t msg_size = param->msg_size;
//     const size_t ttl_len = n_msg * msg_size;
//     const int buddy = peer(param->comm->rank, param->comm->size);
//
//     // we cannot do the fast completion with FENCE or DELIVERY COMPLETE
//     bool do_real_fast = !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL) &&
//                         !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_ORDER) &&
//                         !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
//     double time;
//     rmem_prof_t prof = {.name = "send"};
//     //------------------------------------------------
//     // enqueue the requests
//     rmem_trigr_ptr* trigr = d->trigr;
//     if (gpuMemoryType((void*)d->trigr) != gpuMemoryTypeSystem) {
//         trigr = m_malloc(sizeof(rmem_trigr_ptr) * n_msg);
//     }
//     for (int j = 0; j < n_msg; ++j) {
//         ofi_rma_enqueue(param->mem, d->rma + j, trigr + j, device);
//     }
//     if (gpuMemoryType((void*)d->trigr) != gpuMemoryTypeSystem) {
//         m_gpu_call(gpuMemcpySync((void*)d->trigr, trigr, n_msg * sizeof(rmem_trigr_ptr),
//                                  gpuMemcpyHostToDevice));
//         free((void*)trigr);
//     }
//     // get the start id
//     const int start_id = rmem_get_rand(n_msg);
//     // send a readiness signal
//     if (do_real_fast) {
//         ofi_rmem_start_fast(1, &buddy, param->mem, param->comm);
//     } else {
//         ofi_rmem_start(1, &buddy, param->mem, param->comm);
//     }
//     // compute the time difference
//     ack_offset_sender(ack);
//     if (device == RMEM_AWARE) {
//         m_rmem_prof(prof, time) {
//             for (int j = 0; j < n_msg; ++j) {
//                 const int id = (start_id + j) % n_msg;
//                 ofi_rma_start(param->mem, d->rma + id, RMEM_AWARE);
//             }
//             if (do_real_fast) {
//                 ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
//             } else {
//                 ofi_rmem_complete(1, &buddy, param->mem, param->comm);
//             }
//         }
//     } else {
//         m_rmem_prof(prof, time) {
//             gpu_trigger_op(RMEM_GPU_PUT, start_id, n_msg, d->buf, param->msg_size, d->trigr,
//                            d->stream);
//             m_gpu_call(gpuStreamSynchronize(d->stream));
//             m_verb("rma_run_send_device: rmem_complete");
//             if (do_real_fast) {
//                 ofi_rmem_complete_fast(n_msg, param->mem, param->comm);
//             } else {
//                 ofi_rmem_complete(1, &buddy, param->mem, param->comm);
//             }
//         }
//     }
//     // send the starting time from the profiler
//     ack_send_withtime(ack, &prof.t0);
//     ofi_rma_reset_queue(param->mem);
//     return time;
// }

//--------------------------------------------------------------------------------------------------
static double rma_run_recv_common(run_param_t* param, void* data, void* ack_ptr, rmem_device_t dev,
                           rmem_protocol_t protocol) {
    run_rma_data_t* d = (run_rma_data_t*)data;
    ack_t* ack = (ack_t*)ack_ptr;
    const int n_msg = param->n_msg;
    const size_t msg_size = param->msg_size;
    const size_t ttl_len = n_msg * msg_size;
    const int buddy = peer(param->comm->rank, param->comm->size);

    // we cannot do the fast completion with FENCE or DELIVERY COMPLETE
    bool do_real_fast = !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL) &&
                        !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_ORDER) &&
                        !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
    double time, time_result;
    rmem_prof_t prof = {.name = "recv"};
    //------------------------------------------------
    // wait for the readiness signal
    if (protocol == RMEM_DEFAULT) {
        ack_wait(ack);
        m_rmem_prof(prof, time) {
            ofi_rmem_post(1, &buddy, param->mem, param->comm);
            ofi_rmem_wait(1, &buddy, param->mem, param->comm);
        }
        time_result = time;
    } else if (protocol == RMEM_FAST) {
        if (do_real_fast) {
            ofi_rmem_post_fast(1, &buddy, param->mem, param->comm);
        } else {
            ofi_rmem_post(1, &buddy, param->mem, param->comm);
        }
        // obtain the acknowledgment
        const double offset = ack_offset_recver(ack);
        m_rmem_prof(prof, time) {
            if (do_real_fast) {
                ofi_rmem_wait_fast(n_msg, param->mem, param->comm);
            } else {
                ofi_rmem_wait(1, &buddy, param->mem, param->comm);
            }
        }
        // T sender = t recver + offset => T recver = T sender - offset
        // time elapsed = (T recver -  (T sender - offset))
        struct timespec tsend;
        ack_wait_withtime(ack, &tsend);
        time_result = m_get_wtimes(tsend, prof.t1) + offset;
        m_verb("estimated time of comms = %f vs previously measured one %f, offset is %f",
               time_result, time, offset);
    }
    //------------------------------------------------
    // check the result
    run_test_check(ttl_len, param->mem->buf);
    return time_result;
}
double rma_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_recv_common(param, data, ack_ptr, RMEM_AWARE, RMEM_DEFAULT);
}
double rma_run_recv_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_recv_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_DEFAULT);
}
double rma_fast_run_recv(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_recv_common(param, data, ack_ptr, RMEM_AWARE, RMEM_FAST);
}
double rma_fast_run_recv_gpu(run_param_t* param, void* data, void* ack_ptr) {
    return rma_run_recv_common(param, data, ack_ptr, RMEM_TRIGGER, RMEM_FAST);
}
// double rma_fast_run_recv(run_param_t* param, void* data, void* ack_ptr) {
//     run_rma_data_t* d = (run_rma_data_t*)data;
//     ack_t* ack = (ack_t*)ack_ptr;
//     const int n_msg = param->n_msg;
//     const size_t msg_size = param->msg_size;
//     const size_t ttl_len = n_msg * msg_size;
//     const int buddy = peer(param->comm->rank, param->comm->size);
//
//     // we cannot do the fast completion with FENCE or DELIVERY COMPLETE
//     bool do_real_fast = !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_DELIV_COMPL) &&
//                         !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_ORDER) &&
//                         !(param->comm->prov_mode.rcmpl_mode == M_OFI_RCMPL_FENCE);
//
//     double time;
//     rmem_prof_t prof = {.name = "recv"};
//     //------------------------------------------------
//     if (do_real_fast) {
//         ofi_rmem_post_fast(1, &buddy, param->mem, param->comm);
//     } else {
//         ofi_rmem_post(1, &buddy, param->mem, param->comm);
//     }
//     // obtain the acknowledgment
//     const double offset = ack_offset_recver(ack);
//     m_rmem_prof(prof, time) {
//         if (do_real_fast) {
//             ofi_rmem_wait_fast(n_msg, param->mem, param->comm);
//         } else {
//             ofi_rmem_wait(1, &buddy, param->mem, param->comm);
//         }
//     }
//     // T sender = t recver + offset => T recver = T sender - offset
//     // time elapsed = (T recver -  (T sender - offset))
//     struct timespec tsend;
//     ack_wait_withtime(ack, &tsend);
//     double sync_time = m_get_wtimes(tsend, prof.t1) + offset;
//     m_verb("estimated time of comms = %f vs previously measured one %f, offset is %f", sync_time,
//            time, offset);
//
//     //------------------------------------------------
//     // check the result
//     run_test_check(ttl_len, param->mem->buf);
//     return sync_time;
// }
