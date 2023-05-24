/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include <getopt.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <stdio.h>
#include <stdlib.h>

#include "ofi.h"
#include "rmem_utils.h"
#include "rmem_profile.h"
#include "pmi.h"

#include <omp.h>

#define msg_size 1024
#define n_msg 10


int main(int argc, char** argv) {
    const int nth = 1;//omp_get_max_threads();
    const size_t ttl_len = n_msg * msg_size;
    // create a communicator with as many context as threads
    ofi_comm_t comm;
    comm.n_ctx = nth;
    m_rmem_call(ofi_init(&comm));
    int rank = ofi_get_rank(&comm);

    if (rank % 2 == 0) {
        const int peer = rank + 1;
        // recv buffer
        ofi_rmem_t pmem = {
            .buf = NULL,
            .count = 0,
        };
        ofi_rmem_init(&pmem, &comm);
        // send buffer
        double* src = calloc(ttl_len, sizeof(double));
        for (int i = 0; i < ttl_len; ++i) {
            src[i] = i;
        }

        //-----------------------------------------------------------------------------------------
        // PUT
        ofi_rma_t* put = calloc(n_msg, sizeof(ofi_rma_t));
        for (int i = 0; i < n_msg; ++i) {
            put[i] = (ofi_rma_t){
                .buf = src + i * msg_size,
                .count = msg_size * sizeof(double),
                .disp = i * msg_size * sizeof(double),
                .peer = peer,
            };
            ofi_rma_init(put + i, &pmem, &comm);
        }
        for (int i = 0; i < 10; ++i) {
            PMI_Barrier();
            // start exposure
            ofi_rmem_start(1, &peer, &pmem, &comm);
            for (int i = 0; i < n_msg; ++i) {
                ofi_put_enqueue(put + i, &pmem, 0, &comm);
            }
            ofi_rmem_complete(1, &peer, &pmem, &comm);
        }
        for (int i = 0; i < n_msg; ++i) {
            ofi_rma_free(put + i);
        }
        free(put);
        //-----------------------------------------------------------------------------------------
        // RPUT
        // rmem_rma_t rput = {
        //     .buf = src,
        //     .count = msg_size * sizeof(double),
        //     .peer = peer,
        // };
        // rmem_prof_t time_rput = {.name = "rput"};
        // ofi_rma_init(&rput, &pmem, &comm);
        // m_log("done with RMa init");
        //
        // for (int i = 0; i < 10; ++i) {
        //     m_log("iteration %d",i);
        //     ofi_rmem_start(1, &peer, &pmem, &comm);
        //     ofi_rput_enqueue(&rput, &pmem, 0, &comm);
        //     m_rmem_prof(time_rput) {
        //         ofi_rma_start(&rput);
        //         ofi_rma_wait(&rput);
        //     }
        //     ofi_rmem_complete(1, &peer,&pmem,&comm);
        // }
        // ofi_rma_free(&rput);
        //-----------------------------------------------------------------------------------------
        ofi_rmem_free(&pmem, &comm);
        free(src);
    } else {
        const int peer = rank - 1;
        // allocate the receive buffer
        double* pmem_buf = calloc(ttl_len, sizeof(double));
        ofi_rmem_t pmem = {
            .buf = pmem_buf,
            .count = ttl_len * sizeof(double),
        };
        ofi_rmem_init(&pmem, &comm);

        //-----------------------------------------------------------------------------------------
        // PUT
        rmem_prof_t time_put = {.name = "put"};
        for (int i = 0; i < 10; ++i) {
            m_log(">>>>>> iteration %d <<<<<<",i);
            PMI_Barrier();
            m_rmem_prof(time_put) {
                ofi_rmem_post(1, &peer, &pmem, &comm);
                ofi_rmem_wait(1, &peer, &pmem, &comm);
            }
            for (int i = 0; i < ttl_len; ++i) {
                double res = i;
                if (pmem_buf[i] != res) {
                    m_log("pmem[%d] = %f != %f", i, pmem_buf[i], res);
                }
                pmem_buf[i] = 0.0;
            }
        }
        //-----------------------------------------------------------------------------------------
        // RPUT
        // rmem_prof_t time_rput = {.name = "rput"};
        // for (int i = 0; i < 10; ++i) {
        //     m_rmem_prof(time_rput) {
        //         ofi_rmem_post(1, &peer, &pmem, &comm);
        //         ofi_rmem_wait(1, &peer, &pmem, &comm);
        //     }
        //     for (int i = 0; i < msg_size; ++i) {
        //         double res = i;
        //         m_assert(pmem_buf[i] == res,"pmem[%d] = %f != %f",i,pmem_buf[i],res);
        //     }
        // }
        //-----------------------------------------------------------------------------------------
        ofi_rmem_free(&pmem, &comm);
        free(pmem_buf);
    }
    m_rmem_call(ofi_finalize(&comm));
    return EXIT_SUCCESS;
}
