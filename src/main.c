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
#include "rmem_run.h"
#include "rmem_utils.h"
#include "rmem_argp.h"

#define m_run_time(t) (t ? (t[idx] / imsg) : 0.0)

void print_info(char* foldr_name, char* prov_name, ofi_mode_t* mode) {
    char fname[128];
    snprintf(fname, 128, "%s/rmem.info", foldr_name);
    FILE* file = fopen(fname, "w+");
    m_assert(file, "cannot open %s", fname);

    fprintf(file, "----------------------------------------------------------------\n");
    m_log("----------------------------------------------------------------");
#ifdef GIT_COMMIT
    fprintf(file, "commit: %s\n", GIT_COMMIT);
#else
    fprintf(file, "commit: unknown\n");
#endif

    fprintf(file, "provider: %s\n", prov_name);
    switch (mode->sig_mode) {
        case (M_OFI_SIG_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_SIG_ATOMIC):
            fprintf(file, "\t- signal: ATOMIC\n");
            break;
        case (M_OFI_SIG_CQ_DATA):
            fprintf(file, "\t- signal: CQ DATA\n");
            break;
    };
    switch (mode->rtr_mode) {
        case (M_OFI_RTR_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RTR_ATOMIC):
            fprintf(file, "\t- ready-to-receive: ATOMIC\n");
            break;
        case (M_OFI_RTR_MSG):
            fprintf(file, "\t- ready-to-receive: MSG\n");
            break;
        case (M_OFI_RTR_TAGGED):
            fprintf(file, "\t- ready-to-receive: TAGGED MSG\n");
            break;
    };
    switch (mode->rcmpl_mode) {
        case (M_OFI_RCMPL_NULL):
            m_assert(0, "null is not supported here");
            break;
        case (M_OFI_RCMPL_CQ_DATA):
            fprintf(file, "\t- remote completion: CQ_DATA\n");
            break;
        case (M_OFI_RCMPL_FENCE):
            fprintf(file, "\t- remote completion: FENCE\n");
            break;
        case (M_OFI_RCMPL_REMOTE_CNTR):
            fprintf(file, "\t- remote completion: REMOTE COUNTER\n");
            break;
        case (M_OFI_RCMPL_DELIV_COMPL):
            fprintf(file, "\t- remote completion: DELIVERY COMPLETE\n");
            break;
    };
    fprintf(file, "----------------------------------------------------------------\n");
    fclose(file);
}

int main(int argc, char** argv) {
    //----------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------
    // create a communicator with as many context as threads
    const int nth = 1;  // omp_get_max_threads();
    ofi_comm_t comm;

    // parse arguments
    argp_rmem_t arg_rmem = {0};
    argp_parse(&argp, argc, argv, 0, 0, &arg_rmem);
    // init the comm
    comm.prov_mode = arg_rmem.mode;
    comm.n_ctx = nth;
    m_rmem_call(ofi_init(&comm));

    // allocate the shared mem
    ofi_rmem_t rma_mem = {
        .buf = NULL,
        .count = 0,
    };

    // run parameter
    run_param_t param = {
        .msg_size = 1 << 22,
        .n_msg = 2, 
        .comm = &comm, 
        .mem = &rma_mem
    };

    // allocate the shared mem for the receiver
    if (!is_sender(ofi_get_rank(&comm))) {
        const size_t ttl_len =
            m_min(param.msg_size * param.n_msg * sizeof(int), m_max_size) / sizeof(int);
        // receiver needs the remote buffer
        rma_mem = (ofi_rmem_t){
            .buf = calloc(ttl_len, sizeof(int)),
            .count = ttl_len * sizeof(int),
        };
    }
    ofi_rmem_init(&rma_mem, param.comm);

    //----------------------------------------------------------------------------------------------
    // P2P
    run_time_t p2p_time = {0};
    {
        run_p2p_data_t p2p_data;
        run_t p2p_send = {
            .data = &p2p_data,
            .pre = &p2p_pre_send,
            .run = &p2p_run_send,
            .post = &p2p_post_send,
        };
        run_t p2p_recv = {
            .data = &p2p_data,
            .pre = &p2p_pre_recv,
            .run = &p2p_run_recv,
            .post = &p2p_post_recv,
        };
        run_test(&p2p_send, &p2p_recv, param, &p2p_time);
    }
    //----------------------------------------------------------------------------------------------
    // P2P FAST
    run_time_t p2pf_time= {0};
    {
        run_p2p_data_t p2pf_data;
        run_t p2pf_send = {
            .data = &p2pf_data,
            .pre = &p2p_pre_send,
            .run = &p2p_fast_run_send,
            .post = &p2p_post_send,
        };
        run_t p2pf_recv = {
            .data = &p2pf_data,
            .pre = &p2p_pre_recv,
            .run = &p2p_fast_run_recv,
            .post = &p2p_post_recv,
        };
        run_test(&p2pf_send, &p2pf_recv, param, &p2pf_time);
    }
    //----------------------------------------------------------------------------------------------
    // PUT
    run_time_t put_time = {0};
    {
        run_rma_data_t put_data;
        run_t put_send = {
            .data = &put_data,
            .pre = &put_pre_send,
            .run = &rma_run_send,
            .post = &rma_post,
        };
        run_t put_recv = {
            .data = &put_data,
            .pre = &rma_pre_recv,
            .run = &rma_run_recv,
            .post = &rma_post,
        };
        run_test(&put_send, &put_recv, param, &put_time);
    }
    //----------------------------------------------------------------------------------------------
    // PUT + SIGNAL
    run_time_t psig_time = {0};
    {
        run_rma_data_t psig_data;
        run_t psig_send = {
            .data = &psig_data,
            .pre = &sig_pre_send,
            .run = &rma_run_send,
            .post = &rma_post,
        };
        run_t psig_recv = {
            .data = &psig_data,
            .pre = &rma_pre_recv,
            .run = &sig_run_recv,
            .post = &rma_post,
        };
        //run_test(&psig_send, &psig_recv, param, &psig_time);
    }
    //----------------------------------------------------------------------------------------------
    // PUT FAST
    run_time_t pfast_time = {0};
    {
        run_rma_data_t pfast_data;
        run_t plat_send = {
            .data = &pfast_data,
            .pre = &put_pre_send,
            .run = &rma_fast_run_send,
            .post = &rma_post,
        };
        run_t plat_recv = {
            .data = &pfast_data,
            .pre = &rma_pre_recv,
            .run = &rma_fast_run_recv,
            .post = &rma_post,
        };
        //run_test(&plat_send, &plat_recv, param, &pfast_time);
    }
    //----------------------------------------------------------------------------------------------
    // PUT LATENCY
    run_time_t plat_time = {0};
    {
        run_rma_data_t plat_data;
        run_t plat_send = {
            .data = &plat_data,
            .pre = &put_pre_send,
            .run = &lat_run_send,
            .post = &rma_post,
        };
        run_t plat_recv = {
            .data = &plat_data,
            .pre = &rma_pre_recv,
            .run = &lat_run_recv,
            .post = &rma_post,
        };
        //run_test(&plat_send, &plat_recv, param, &plat_time);
    }
    //----------------------------------------------------------------------------------------------
    // free
    ofi_rmem_free(&rma_mem, param.comm);
    free(rma_mem.buf);

    //----------------------------------------------------------------------------------------------
    if (!is_sender(comm.rank)) {
        // get a unique folder for the results
        char fullname[128];
        char foldr_name[64];
        struct stat st = {0};
        int fid_cntr = 0;
        do {
            sprintf(foldr_name, "data_%d", fid_cntr++);
        } while (!stat(foldr_name, &st));
        mkdir(foldr_name, 0770);
        m_verb("found folder: %s",foldr_name);

        // get some 101 info
        print_info(foldr_name, ofi_name(&comm),&comm.prov_mode);

        //------------------------------------------------------------------------------------------
        // save the results per msg size
        const int n_size = log10(param.msg_size) / log10(2.0) + 1;
        const int n_msg = log10(param.n_msg) / log10(2.0) + 1;
        for (int imsg = 1; imsg <= param.n_msg; imsg *= 2) {
            const int idx_msg = log10(imsg) / log10(2.0);
            char fullname[128];
            snprintf(fullname, 128, "%s/msg%d_%s.txt", foldr_name, imsg, ofi_name(&comm));
            // get a unique FID for the results
            FILE* file = fopen(fullname, "w+");
            m_assert(file, "cannot open %s", fullname);

            m_log("--------------- %d MSGs ---------------",imsg);
            // for each message size
            for (size_t msg_size = 1; msg_size <= param.msg_size; msg_size *= 2) {
                if (imsg * msg_size * sizeof(int) > m_max_size) {
                    break;
                }
                const int idx_size = log10(msg_size) / log10(2.0);
                // get the idx
                int idx = idx_msg * n_size + idx_size;
                // load the results
                const double ti_p2p = m_run_time(p2p_time.avg);
                const double ci_p2p = m_run_time(p2p_time.ci);
                const double ti_put = m_run_time(put_time.avg);
                const double ci_put = m_run_time(put_time.ci);
                const double ti_p2pf = m_run_time(p2pf_time.avg);
                const double ci_p2pf = m_run_time(p2pf_time.ci);
                const double ti_psig = m_run_time(psig_time.avg);
                const double ci_psig = m_run_time(psig_time.ci);
                const double ti_plat = m_run_time(plat_time.avg);
                const double ci_plat = m_run_time(plat_time.ci);
                const double ti_fast = m_run_time(pfast_time.avg);
                const double ci_fast = m_run_time(pfast_time.ci);
                m_log(
                    "time/msg (%ld B/msg - %d msgs):\n"
                    "\tP2P       = %f +-[%f]\n"
                    "\tPUT       = %f +-[%f] (ratio = %f)\n"
                    "\tPUT FAST  = %f +-[%f] (ratio = %f)\n"
                    "\tPUT + SIG = %f +-[%f] (ratio = %f)\n"
                    "\tPUT LAT   = %f +-[%f] (ratio = %f)\n"
                    "\tP2P FAST  = %f +-[%f] (ratio = %f)\n",
                    msg_size * sizeof(int), imsg, ti_p2p, ci_p2p, ti_put, ci_put, ti_put / ti_p2p,
                    ti_fast, ci_fast, ti_fast / ti_p2p, ti_psig, ci_psig, ti_psig / ti_p2p, ti_plat,
                    ci_plat, ti_plat / ti_p2p, ti_p2pf, ci_p2pf, ti_p2pf / ti_p2p);
                // write to csv
                fprintf(file, "%ld,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", msg_size * sizeof(int),
                        ti_p2p, ti_put, ti_fast, ti_psig, ti_plat, ti_p2pf, ci_p2p, ci_put, ci_fast,
                        ci_psig, ci_plat, ci_p2pf);
                // bump the index
                idx++;
            }
            fclose(file);
        }
        //------------------------------------------------------------------------------------------
        // save the results per number of msgs
        for (size_t msg_size = 1; msg_size <= param.msg_size; msg_size *= 2) {
            const int idx_size = log10(msg_size) / log10(2.0);
            char fullname[128];
            snprintf(fullname, 128, "%s/size%ld_%s.txt", foldr_name, msg_size, ofi_name(&comm));
            // get a unique FID for the results
            FILE* file = fopen(fullname, "w+");
            m_assert(file, "cannot open %s", fullname);

            // for each message size
            for (int imsg = 1; imsg <= param.n_msg; imsg *= 2) {
                if (imsg * msg_size * sizeof(int) > m_max_size) {
                    break;
                }
                const int idx_msg = log10(imsg) / log10(2.0);
                // get the idx
                int idx = idx_msg * n_size + idx_size;
                // load the results
                const double ti_p2p = m_run_time(p2p_time.avg);
                const double ci_p2p = m_run_time(p2p_time.ci);
                const double ti_put = m_run_time(put_time.avg);
                const double ci_put = m_run_time(put_time.ci);
                const double ti_p2pf = m_run_time(p2pf_time.avg);
                const double ci_p2pf = m_run_time(p2pf_time.ci);
                const double ti_psig = m_run_time(psig_time.avg);
                const double ci_psig = m_run_time(psig_time.ci);
                const double ti_plat = m_run_time(plat_time.avg);
                const double ci_plat = m_run_time(plat_time.ci);
                const double ti_fast = m_run_time(pfast_time.avg);
                const double ci_fast = m_run_time(pfast_time.ci);
                // write to csv
                fprintf(file, "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", imsg, ti_p2p, ti_put,
                        ti_fast, ti_psig, ti_plat, ti_p2pf, ci_p2p, ci_put, ci_fast, ci_psig,
                        ci_plat, ci_p2pf);
                // bump the index
                idx++;
            }
            fclose(file);
        }
    }
    if (p2p_time.avg) free(p2p_time.avg);
    if (p2p_time.ci) free(p2p_time.ci);
    if (p2pf_time.avg) free(p2pf_time.avg);
    if (p2pf_time.ci) free(p2pf_time.ci);
    if (put_time.avg) free(put_time.avg);
    if (put_time.ci) free(put_time.ci);
    if (psig_time.avg) free(psig_time.avg);
    if (psig_time.ci) free(psig_time.ci);
    if (plat_time.avg) free(plat_time.avg);
    if (plat_time.ci) free(plat_time.ci);
    if (pfast_time.avg) free(pfast_time.avg);
    if (pfast_time.ci) free(pfast_time.ci);

    m_rmem_call(ofi_finalize(&comm));
    return EXIT_SUCCESS;
}
