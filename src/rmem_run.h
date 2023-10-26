/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef RMEM_RUN_H_
#define RMEM_RUN_H_

#include "ofi.h"

#define m_max_ttl_size (1 << 24)
#define m_min_msg_size (1 << 20)

#define m_min_msg        512
#define m_min_size       1
#define m_msg_idx(imsg)  ((log10(imsg) - log10(m_min_msg)) / log10(2))
#define m_size_idx(imsg) ((log10(imsg) - log10(m_min_size)) / log10(2))
#define m_msg_size(n, size, type)                                             \
    ({                                                                        \
        size_t n_msg_size_limit = m_max(m_max_ttl_size, m_min_msg_size * n);  \
        m_min(n* size * sizeof(type), n_msg_size_limit) / (sizeof(type) * n); \
    })

static int is_sender(const int rank) { return !(rank % 2); }
static int peer(const int rank, const int size) {
    if (is_sender(rank)) {
        return (rank + 1) % size;
    } else {
        return (rank - 1) % size;
    }
};
typedef struct {
    double* avg;
    double* ci;
} run_time_t;

typedef struct {
    int n_msg;        // number of messages
    size_t msg_size;  // size of a message
    ofi_rmem_t* mem;
    ofi_comm_t* comm;
} run_param_t;

typedef struct {
    void* data;
    void (*pre)(run_param_t* param, void*);
    double (*run)(run_param_t* param, void*, void*);
    void (*post)(run_param_t* param, void*);
} run_t;

typedef struct {
    ofi_p2p_t* p2p;
    int* buf;
    gpuStream_t stream;
} run_p2p_data_t;

typedef struct {
    ofi_rma_t* rma;  // array of op
    int* buf;
    rmem_trigr_ptr* trigr;
    gpuStream_t stream;
} run_rma_data_t;

void run_test(run_t* sender, run_t* recver, run_param_t param, run_time_t* timings);

//--------------------------------------------------------------------------------------------------
// point to point
void p2p_pre_send(run_param_t* param, void* data);
void p2p_pre_recv(run_param_t* param, void* data);
void p2p_post_send(run_param_t* param, void* data);
void p2p_post_recv(run_param_t* param, void* data);
double p2p_run_send(run_param_t* param, void* data, void* ack);
double p2p_run_recv(run_param_t* param, void* data, void* ack);
double p2p_run_send_gpu(run_param_t* param, void* data, void* ack);
double p2p_run_recv_gpu(run_param_t* param, void* data, void* ack);
double p2p_fast_run_send(run_param_t* param, void* data, void* ack);
double p2p_fast_run_recv(run_param_t* param, void* data, void* ack);

//--------------------------------------------------------------------------------------------------
// PRE
//----------------
// send
void put_pre_send(run_param_t* param, void* data);
// recv
void put_pre_recv(run_param_t* param, void* data);

//----------------
// POST
//----------------
void rma_post(run_param_t* param, void* data);

//----------------
// RUN
//----------------
// send
double rma_run_send(run_param_t* param, void* data, void* ack);
double rma_fast_run_send(run_param_t* param, void* data, void* ack);
double rma_run_send_gpu(run_param_t* param, void* data, void* ack);
double rma_fast_run_send_gpu(run_param_t* param, void* data, void* ack);
// recv
double rma_run_recv(run_param_t* param, void* data, void* ack);
double rma_fast_run_recv(run_param_t* param, void* data, void* ack);
double rma_run_recv_gpu(run_param_t* param, void* data, void* ack);
double rma_fast_run_recv_gpu(run_param_t* param, void* data, void* ack);

#endif
