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

//#define msg_size 1024
#define n_msg 1
#define max_size (1<<18)
#define n_measure 10
#define n_warmup 5
#define retry_threshold 0.05
#define retry_max 1

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
    // create the send/recv for retry
    int retry = 0;
    ofi_p2p_t p2p_retry = {
        .buf = &retry,
        .count = sizeof(retry),
        .tag = n_msg + 1,
    };
    // no peer is needed to create the request
    ofi_p2p_create(&p2p_retry, &comm);

    //----------------------------------------------------------------------------------------------
    for (size_t msg_size = 1; msg_size < max_size; msg_size *= 2) {
        PMI_Barrier();
        const size_t ttl_len = n_msg * msg_size;
        if (rank % 2 == 0) {
            //======================================================================================
            // SENDER
            //======================================================================================
            // get peer and retry comm
            const int peer = rank + 1;
            p2p_retry.peer = peer;

            // send buffer
            ofi_rmem_t pmem = {
                .buf = NULL,
                .count = 0,
            };
            ofi_rmem_init(&pmem, &comm);
            int* src = calloc(ttl_len, sizeof(int));
            for (int i = 0; i < ttl_len; ++i) {
                src[i] = i + 1;
            }
            //--------------------------------------------------------------------------------------
            // POINT TO POINT
            ofi_p2p_t* send = calloc(n_msg, sizeof(ofi_rma_t));
            for (int i = 0; i < n_msg; ++i) {
                send[i] = (ofi_p2p_t){
                    .buf = src + i * msg_size,
                    .count = msg_size * sizeof(int),
                    .peer = peer,
                    .tag = i,
                };
                ofi_p2p_create(send + i, &comm);
            }
            retry = 1;
            while (retry) {
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
                ofi_recv_enqueue(&p2p_retry,0,&comm);
                ofi_p2p_wait(&p2p_retry);
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_p2p_free(send + i);
            }
            free(send);

            //--------------------------------------------------------------------------------------
            // PUT
            ofi_rma_t* put = calloc(n_msg, sizeof(ofi_rma_t));
            ofi_drma_t* dput = calloc(n_msg, sizeof(ofi_drma_t));
            for (int i = 0; i < n_msg; ++i) {
                put[i] = (ofi_rma_t){
                    .buf = src + i * msg_size,
                    .count = msg_size * sizeof(int),
                    .disp = i * msg_size * sizeof(int),
                    .peer = peer,
                };
                // ofi_rma_init(put + i, &pmem, &comm);
            }
            retry = 1;
            while (retry) {
                for (int it = -n_warmup; it < n_measure; ++it) {
                    for (int j = 0; j < n_msg; ++j) {
                        ofi_drma_t* c_dput = dput + j;
                        ofi_put_enqueue(put + j, &pmem, 0, &comm, &(c_dput));
                    }
                    PMI_Barrier();  // start exposure
                    ofi_rmem_start(1, &peer, &pmem, &comm);
                    for (int j = 0; j < n_msg; ++j) {
                        ofi_rma_start(dput + j);
                    }
                    ofi_rmem_complete(1, &peer, &pmem, &comm);
                }
                ofi_recv_enqueue(&p2p_retry, 0, &comm);
                ofi_p2p_wait(&p2p_retry);
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_rma_free(put + i, dput + i);
            }
            free(put);
            free(dput);
            //--------------------------------------------------------------------------------------
            // PUT + SIGNAL
            ofi_rma_t* psig = calloc(n_msg, sizeof(ofi_rma_t));
            ofi_drma_t* dpsig = calloc(n_msg, sizeof(ofi_drma_t));
            for (int i = 0; i < n_msg; ++i) {
                psig[i] = (ofi_rma_t){
                    .buf = src + i * msg_size,
                    .count = msg_size * sizeof(int),
                    .disp = i * msg_size * sizeof(int),
                    .peer = peer,
                };
            }
            retry = 1;
            while (retry) {
                for (int it = -n_warmup; it < n_measure; ++it) {
                    for (int j = 0; j < n_msg; ++j) {
                        ofi_drma_t* c_dpsig = dpsig + j;
                        ofi_put_signal_enqueue(psig + j, &pmem, 0, &comm,&c_dpsig);
                    }
                    PMI_Barrier();  // start exposure
                    ofi_rmem_start(1, &peer, &pmem, &comm);
                    for (int j = 0; j < n_msg; ++j) {
                        ofi_rma_start(dpsig + j);
                    }
                    ofi_rmem_complete(1, &peer, &pmem, &comm);
                }
                ofi_recv_enqueue(&p2p_retry, 0, &comm);
                ofi_p2p_wait(&p2p_retry);
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_rma_free(psig + i,dpsig + i);
            }
            free(psig);
            free(dpsig);
            //--------------------------------------------------------------------------------------
            // RPUT
            // rmem_rma_t rput = {
            //     .buf = src,
            //     .count = msg_size * sizeof(int),
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
            p2p_retry.peer = peer;

            // allocate the receive buffer
            int* pmem_buf = calloc(ttl_len, sizeof(int));
            ofi_rmem_t pmem = {
                .buf = pmem_buf,
                .count = ttl_len * sizeof(int),
            };
            ofi_rmem_init(&pmem, &comm);

            //--------------------------------------------------------------------------------------
            // POINT-TO-POINT
            ofi_p2p_t* recv = calloc(n_msg, sizeof(ofi_rma_t));
            for (int i = 0; i < n_msg; ++i) {
                recv[i] = (ofi_p2p_t){
                    .buf = pmem_buf + i * msg_size,
                    .count = msg_size * sizeof(int),
                    .peer = peer,
                    .tag = i,
                };
                ofi_p2p_create(recv + i, &comm);
            }

            // profile stuff
            rmem_prof_t prof_recv = {.name = "recv"};
            double tavg_p2p;
            double ci_p2p;

            // let's go
            retry = 1;
            while (retry) {
                double time[n_measure];
                for (int it = -n_warmup; it < n_measure; ++it) {
                    PMI_Barrier();
                    m_rmem_prof(prof_recv, time[(it >= 0) ? it : 0]) {
                        for (int j = 0; j < n_msg; ++j) {
                            ofi_recv_enqueue(recv + j, 0, &comm);
                        }
                        for (int j = 0; j < n_msg; ++j) {
                            ofi_p2p_wait(recv + j);
                        }
                    }
                    // check the result
                    for (int i = 0; i < ttl_len; ++i) {
                        int res = i + 1;
                        if (pmem_buf[i] != res) {
                            m_log("pmem[%d] = %d != %d", i, pmem_buf[i], res);
                        }
                        pmem_buf[i] = 0.0;
                    }
                }
                // get the CI + the retry
                rmem_get_ci(n_measure, time, &tavg_p2p, &ci_p2p);
                m_verb("SEND: avg = %f, CI = %f, ratio = %f vs %f retry = %d/%d", tavg_p2p, ci_p2p,
                       ci_p2p / tavg_p2p, retry_threshold, retry, retry_max);
                if (retry > retry_max || (ci_p2p / tavg_p2p) < retry_threshold) {
                    retry = 0;
                } else {
                    retry++;
                }
                // send the information to the sender side
                ofi_send_enqueue(&p2p_retry, 0, &comm);
                ofi_p2p_wait(&p2p_retry);
            }
            for (int i = 0; i < n_msg; ++i) {
                ofi_p2p_free(recv + i);
            }
            free(recv);

            //--------------------------------------------------------------------------------------
            // PUT
            double tavg_put;
            double ci_put;
            rmem_prof_t prof_put = {.name = "put"};
            retry = 1;
            while (retry) {
                double time[n_measure];
                for (int it = -n_warmup; it < n_measure; ++it) {
                    PMI_Barrier();
                    m_rmem_prof(prof_put, time[(it >= 0) ? it : 0]) {
                        ofi_rmem_post(1, &peer, &pmem, &comm);
                        ofi_rmem_wait(1, &peer, &pmem, &comm);
                    }
                    // check the results
                    for (int i = 0; i < ttl_len; ++i) {
                        int res = i + 1;
                        m_assert(pmem_buf[i] == res,"pmem[%d] = %d != %d", i, pmem_buf[i], res);
                        if (pmem_buf[i] != res) {
                            m_log("pmem[%d] = %d != %d", i, pmem_buf[i], res);
                        }
                        pmem_buf[i] = 0.0;
                    }
                }
                // get the CI + the retry
                rmem_get_ci(n_measure, time, &tavg_put, &ci_put);
                if (retry > retry_max || (ci_put / tavg_put) < retry_threshold) {
                    retry = 0;
                } else {
                    retry++;
                }
                m_verb("PUT: avg = %f, CI = %f, ratio = %f vs %f retry = %d/%d", tavg_put, ci_put,
                       ci_put / tavg_put, retry_threshold, retry, retry_max);
                // send the information to the sender side
                ofi_send_enqueue(&p2p_retry, 0, &comm);
                ofi_p2p_wait(&p2p_retry);
            }
            //--------------------------------------------------------------------------------------
            // PUT + SIGNAL
            double tavg_psig;
            double ci_psig;
            rmem_prof_t prof_psig = {.name = "put-signal"};
            retry = 1;
            while (retry) {
                double time[n_measure];
                for (int it = -n_warmup; it < n_measure; ++it) {
                    PMI_Barrier();
                    m_rmem_prof(prof_psig, time[(it >= 0) ? it : 0]) {
                        ofi_rmem_post(1, &peer, &pmem, &comm);
                        ofi_rmem_wait(1, &peer, &pmem, &comm);
                    }
                    // check the results
                    for (int i = 0; i < ttl_len; ++i) {
                        int res = i + 1;
                        if (pmem_buf[i] != res) {
                            m_log("pmem[%d] = %d != %d", i, pmem_buf[i], res);
                        }
                        pmem_buf[i] = 0.0;
                    }
                }
                // get the CI + the retry
                rmem_get_ci(n_measure, time, &tavg_psig, &ci_psig);
                if (retry > retry_max || (ci_psig / tavg_psig) < retry_threshold) {
                    retry = 0;
                } else {
                    retry++;
                }
                m_verb("PUT+SIG: avg = %f, CI = %f, ratio = %f vs %f retry = %d/%d", tavg_psig, ci_psig,
                       ci_psig / tavg_psig, retry_threshold, retry, retry_max);
                // send the information to the sender side
                ofi_send_enqueue(&p2p_retry, 0, &comm);
                ofi_p2p_wait(&p2p_retry);
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
            //         int res = i + 1;
            //         m_assert(pmem_buf[i] == res,"pmem[%d] = %f != %f",i,pmem_buf[i],res);
            //     }
            // }
            //--------------------------------------------------------------------------------------
            ofi_rmem_free(&pmem, &comm);
            free(pmem_buf);
            //--------------------------------------------------------------------------------------
            // display the results
            m_log(
                "time (%ld B - %d msgs):\n"
                "\tP2P       = %f +-[%f]\n"
                "\tPUT       = %f +-[%f] (ratio = %f)\n"
                "\tPUT + SIG = %f +-[%f] (ratio = %f)",
                ttl_len * sizeof(int), n_msg, tavg_p2p, ci_p2p, tavg_put, ci_put,
                tavg_put / tavg_p2p, tavg_psig, ci_psig, tavg_psig / tavg_p2p);
            // write to csv
            FILE* file = fopen(fullname, "a");
            m_assert(file, "file must be open");
            fprintf(file, "%ld,%f,%f,%f,%f,%f,%f\n", ttl_len * sizeof(int), tavg_p2p, tavg_put,
                    tavg_psig, ci_p2p, ci_put, ci_psig);
            fclose(file);
            //--------------------------------------------------------------------------------------
        }
        ofi_p2p_free(&p2p_retry);
    }
    m_rmem_call(ofi_finalize(&comm));
    return EXIT_SUCCESS;
}
