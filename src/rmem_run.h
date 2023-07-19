/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef RMEM_RUN_H_
#define RMEM_RUN_H_

#define m_max_size (1<<24)

#include "ofi.h"

static int is_sender(const int rank){
    return !(rank%2);
}
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
    ofi_comm_t* comm;
    int n_msg;     // number of messages
    int msg_size;  // size of a message
} run_param_t;

typedef struct {
    void* data;
    void (*pre)(run_param_t* param, void*);
    double (*run)(run_param_t* param, void*);
    void (*post)(run_param_t* param, void*);
} run_t;

typedef struct {
    ofi_p2p_t* p2p;
    int* buf;
} run_p2p_data_t;

typedef struct {
    ofi_rmem_t mem; // remote mem
    ofi_rma_t* rma; // array of op
    int* buf;
} run_rma_data_t;

void run_test(run_t* sender, run_t* recver, run_param_t param, run_time_t* timings);

//--------------------------------------------------------------------------------------------------
// point to point
void p2p_pre(run_param_t* param, void* data);
void p2p_post(run_param_t* param, void* data);
double p2p_run_send(run_param_t* param, void* data);
double p2p_run_recv(run_param_t* param, void* data);

//--------------------------------------------------------------------------------------------------
// PRE
//----------------
// send
void put_pre_send(run_param_t* param, void* data);
void sig_pre_send(run_param_t* param, void* data);
// recv
void rma_pre_recv(run_param_t* param, void* data);

//----------------
// POST
//----------------
void rma_post(run_param_t* param, void* data);

//----------------
// RUN
//----------------
// send
double rma_run_send(run_param_t* param, void* data);
double rma_fast_run_send(run_param_t* param, void* data);
double lat_run_send(run_param_t* param, void* data);
// recv
double rma_run_recv(run_param_t* param, void* data);
double rma_fast_run_recv(run_param_t* param, void* data);
double sig_run_recv(run_param_t* param, void* data);
double lat_run_recv(run_param_t* param, void* data);

#endif