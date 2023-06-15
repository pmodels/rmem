/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "ofi.h"
#include "pmi.h"
#include "rmem_profile.h"
#include "rmem_utils.h"

//#define msg_size 1024
#define n_msg 1
#define max_size (1<<18)
#define n_measure 150
#define n_warmup 5

int main(int argc, char** argv) {
    const int nth = 1;  // omp_get_max_threads();
    // create a communicator with as many context as threads
    ofi_comm_t comm;
    comm.n_ctx = nth;
    m_rmem_call(ofi_init(&comm));
    int rank = ofi_get_rank(&comm);

    //----------------------------------------------------------------------------------------------
    // test and create the dir if needed
    char fullname[128];
    if (!(rank % 2 == 0)) {
        char foldr_name[64] = "data";
        sprintf(fullname, "%s/rmem_%d.txt", foldr_name, n_msg);
        struct stat st = {0};
        if (stat(foldr_name, &st) == -1) {
            mkdir(foldr_name, 0770);
        }
        FILE* file = fopen(fullname, "w+");
        m_assert(file, "cannot open %s", fullname);
        fclose(file);
    }

    //----------------------------------------------------------------------------------------------
    for (size_t msg_size = 1; msg_size < max_size; msg_size *= 2) {
        PMI_Barrier();
        // m_log("===========================================================");
        const size_t ttl_len = n_msg * msg_size;
        if (rank % 2 == 0) {
            //======================================================================================
            // SENDER
            //======================================================================================
            const int peer = rank + 1;
            // send buffer
            ofi_rmem_t pmem = {
                .buf = NULL,
                .count = 0,
            };
            ofi_rmem_init(&pmem, &comm);
            double* src = calloc(ttl_len, sizeof(double));
            for (int i = 0; i < ttl_len; ++i) {
                src[i] = i;
            }
            //--------------------------------------------------------------------------------------
            // POINT TO POINT
            ofi_p2p_t* send = calloc(n_msg, sizeof(ofi_rma_t));
            for (int i = 0; i < n_msg; ++i) {
                send[i] = (ofi_p2p_t){
                    .buf = src + i * msg_size,
                    .count = msg_size * sizeof(double),
                    .peer = peer,
                };
                ofi_p2p_create(send + i, &comm);
            }
            for (int it = -n_warmup; it < n_measure; ++it) {
                PMI_Barrier();
                // start exposure
                for (int j = 0; j < n_msg; ++j) {
                    ofi_send_enqueue(send + j, 0, &comm);
                }
                for (int j = 0; j < n_msg; ++j) {
                    ofi_p2p_wait(send + j);
                }
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_p2p_free(send + i);
            }
            free(send);

            //--------------------------------------------------------------------------------------
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
            for (int it = -n_warmup; it < n_measure; ++it) {
                PMI_Barrier();  // start exposure
                ofi_rmem_start(1, &peer, &pmem, &comm);
                for (int j = 0; j < n_msg; ++j) {
                    ofi_put_enqueue(put + j, &pmem, 0, &comm);
                }
                ofi_rmem_complete(1, &peer, &pmem, &comm);
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_rma_free(put + i);
            }
            free(put);
            //--------------------------------------------------------------------------------------
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
            //--------------------------------------------------------------------------------------
            ofi_rmem_free(&pmem, &comm);
            free(src);
        } else {
            //======================================================================================
            // RECIEVER
            //======================================================================================
            const int peer = rank - 1;
            // allocate the receive buffer
            double* pmem_buf = calloc(ttl_len, sizeof(double));
            ofi_rmem_t pmem = {
                .buf = pmem_buf,
                .count = ttl_len * sizeof(double),
            };
            ofi_rmem_init(&pmem, &comm);

            //--------------------------------------------------------------------------------------
            // POINT-TO-POINT
            ofi_p2p_t* recv = calloc(n_msg, sizeof(ofi_rma_t));
            rmem_prof_t prof_recv = {.name = "recv"};
            for (int i = 0; i < n_msg; ++i) {
                recv[i] = (ofi_p2p_t){
                    .buf = pmem_buf + i * msg_size,
                    .count = msg_size * sizeof(double),
                    .peer = peer,
                };
                ofi_p2p_create(recv + i, &comm);
            }
            double t_p2p[n_measure];
            for (int it = -n_warmup; it < n_measure; ++it) {
                PMI_Barrier();
                m_rmem_prof(prof_recv,t_p2p[(it >= 0) ? it : 0]) {
                    for (int j = 0; j < n_msg; ++j) {
                        ofi_recv_enqueue(recv + j, 0, &comm);
                    }
                    for (int j = 0; j < n_msg; ++j) {
                        ofi_p2p_wait(recv + j);
                    }
                }
                // check the result
                for (int i = 0; i < ttl_len; ++i) {
                    double res = i;
                    if (pmem_buf[i] != res) {
                        m_log("pmem[%d] = %f != %f", i, pmem_buf[i], res);
                    }
                    pmem_buf[i] = 0.0;
                }
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_p2p_free(recv + i);
            }
            free(recv);

            //--------------------------------------------------------------------------------------
            // PUT
            //m_log("============= PUT (%d at once) =============", n_msg);
            double t_put[n_measure];
            rmem_prof_t prof_put = {.name = "put"};
            for (int it = -n_warmup; it < n_measure; ++it) {
                PMI_Barrier();
                m_rmem_prof(prof_put, t_put[(it >= 0) ? it : 0]) {
                    ofi_rmem_post(1, &peer, &pmem, &comm);
                    ofi_rmem_wait(1, &peer, &pmem, &comm);
                }
                // check the results
                for (int i = 0; i < ttl_len; ++i) {
                    double res = i;
                    if (pmem_buf[i] != res) {
                        m_log("pmem[%d] = %f != %f", i, pmem_buf[i], res);
                    }
                    pmem_buf[i] = 0.0;
                }
            }
            //--------------------------------------------------------------------------------------
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
            //--------------------------------------------------------------------------------------
            ofi_rmem_free(&pmem, &comm);
            free(pmem_buf);
            //--------------------------------------------------------------------------------------
            // get the right std
            double tavg_p2p = 0.0;
            double tavg_put = 0.0;
            for (int i = 0; i < n_measure; ++i) {
                    tavg_p2p += t_p2p[i] / n_measure;
                    tavg_put += t_put[i] / n_measure;
            }
            double tstd_p2p = 0.0;
            double tstd_put = 0.0;
            for (int i = 0; i < n_measure; ++i) {
                    tstd_p2p += pow(t_p2p[i] - tavg_p2p, 2);
                    tstd_put += pow(t_put[i] - tavg_put, 2);
            }
            // get the CI
            const double t_nu_val = t_nu_interp(n_measure);
            const double s_p2p = sqrt(tstd_p2p / (n_measure - 1));
            const double s_put = sqrt(tstd_put / (n_measure - 1));
            const double ci_p2p = s_p2p * t_nu_val * sqrt(1.0 / n_measure);
            const double ci_put = s_put * t_nu_val * sqrt(1.0 / n_measure);

            // display the results
            m_log("time (%ld B - %d msgs): P2P = %f +-[%f] - PUT = %f +-[%f] - ratio = %f",
                  ttl_len * sizeof(double), n_msg, tavg_p2p, ci_p2p, tavg_put, ci_put,
                  tavg_put / tavg_p2p);
            // write to csv
            FILE* file = fopen(fullname, "a");
            m_assert(file, "file must be open");
            fprintf(file, "%ld,%f,%f,%f,%f\n", ttl_len * sizeof(double), tavg_p2p, tavg_put, ci_p2p,
                    ci_put);
            fclose(file);
            //--------------------------------------------------------------------------------------
        }
    }
    m_rmem_call(ofi_finalize(&comm));
    return EXIT_SUCCESS;
}
