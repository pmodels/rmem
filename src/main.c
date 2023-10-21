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

#ifdef HAVE_CUDA
    fprintf(file, "CUDA GPU\n");
#elif defined(HAVE_HIP)
    fprintf(file, "HIP GPU\n");
#else
    fprintf(file, "NO GPU\n");
#endif
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
        .msg_size = 1 << 20,
        .n_msg = 64, 
        .comm = &comm, 
        .mem = &rma_mem
    };

    // allocate the shared mem for the receiver
    if (!is_sender(ofi_get_rank(&comm))) {
        const size_t ttl_len = m_msg_size(param.n_msg, param.msg_size, int) * param.n_msg;
        m_verb("sender memory is %lu", ttl_len);
#if (M_HAVE_GPU)
        // receiver needs the remote buffer
        rma_mem = (ofi_rmem_t){
            .buf = NULL,
            .count = ttl_len * sizeof(int),
        };
        m_gpu_call(gpuMalloc((void**)&rma_mem.buf, ttl_len * sizeof(int)));
#else
        // receiver needs the remote buffer
        rma_mem = (ofi_rmem_t){
            .buf = calloc(ttl_len, sizeof(int)),
            .count = ttl_len * sizeof(int),
        };

#endif
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
    // P2P
    run_time_t p2pgpu_time = {0};
    {
        run_p2p_data_t p2p_data;
        run_t p2p_send = {
            .data = &p2p_data,
            .pre = &p2p_pre_send,
            .run = &p2p_run_send_gpu,
            .post = &p2p_post_send,
        };
        run_t p2p_recv = {
            .data = &p2p_data,
            .pre = &p2p_pre_recv,
            .run = &p2p_run_recv_gpu,
            .post = &p2p_post_recv,
        };
        run_test(&p2p_send, &p2p_recv, param, &p2pgpu_time);
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
            .pre = &put_pre_recv,
            .run = &rma_run_recv,
            .post = &rma_post,
        };
        run_test(&put_send, &put_recv, param, &put_time);
    }
    //----------------------------------------------------------------------------------------------
    // PUT + TRIGGER FROM GPU
    run_time_t pgpu_time ={0};
    {
        run_rma_data_t put_data;
        run_t put_send = {
            .data = &put_data,
            .pre = &put_pre_send,
            .run = &rma_run_send_gpu,
            .post = &rma_post,
        };
        run_t put_recv = {
            .data = &put_data,
            .pre = &put_pre_recv,
            .run = &rma_run_recv_gpu,
            .post = &rma_post,
        };
        run_test(&put_send, &put_recv, param, &pgpu_time);
    }
    //----------------------------------------------------------------------------------------------
    // PUT FAST
    run_time_t pfast_time = {0};
    {
        run_rma_data_t pfast_data;
        run_t pfast_send = {
            .data = &pfast_data,
            .pre = &put_pre_send,
            .run = &rma_fast_run_send,
            .post = &rma_post,
        };
        run_t pfast_recv = {
            .data = &pfast_data,
            .pre = &put_pre_recv,
            .run = &rma_fast_run_recv,
            .post = &rma_post,
        };
        run_test(&pfast_send, &pfast_recv, param, &pfast_time);
    }
    //----------------------------------------------------------------------------------------------
    // free
    ofi_rmem_free(&rma_mem, param.comm);
#if (M_HAVE_GPU)
    m_gpu_call(gpuFree(rma_mem.buf));
#else
    free(rma_mem.buf);
#endif

    //----------------------------------------------------------------------------------------------
    // get a unique folder for the results
    char foldr_name[64];
    int fid_cntr = -1;
    ofi_p2p_t p2p_fid_cntr = {
        .peer = peer(comm.rank, comm.size),
        .buf = &fid_cntr,
        .count = sizeof(int),
        .tag = param.n_msg + 2,
    };
    if (!is_sender(comm.rank)) {
        struct stat st = {0};
        do {
            sprintf(foldr_name, "data_%d", ++fid_cntr);
        } while (!stat(foldr_name, &st));
        mkdir(foldr_name, 0770);
        m_verb("found folder: %s", foldr_name);

        // get some 101 info
        print_info(foldr_name, ofi_name(&comm), &comm.prov_mode);
        // send the info to the other rank
        ofi_send_init(&p2p_fid_cntr, 0, &comm);
        ofi_p2p_start(&p2p_fid_cntr);
        ofi_p2p_wait(&p2p_fid_cntr);
        ofi_p2p_free(&p2p_fid_cntr);

    } else {
        ofi_recv_init(&p2p_fid_cntr, 0, &comm);
        ofi_p2p_start(&p2p_fid_cntr);
        ofi_p2p_wait(&p2p_fid_cntr);
        ofi_p2p_free(&p2p_fid_cntr);
        sprintf(foldr_name, "data_%d", fid_cntr);
    }
    PMI_Barrier();
    //------------------------------------------------------------------------------------------
    // save the results per msg size
    const int n_size = log10(param.msg_size) / log10(2.0) + 1;
    const int n_msg = log10(param.n_msg) / log10(2.0) + 1;
    for (int imsg = 1; imsg <= param.n_msg; imsg *= 2) {
        const int idx_msg = log10(imsg) / log10(2.0);
        char fullname[128];
        snprintf(fullname, 128, "%s/r%d_msg%d_%s.txt", foldr_name, comm.rank, imsg,
                 ofi_name(&comm));

        m_verb("filename is %s", fullname);
        // get a unique FID for the results
        FILE* file = fopen(fullname, "w+");
        m_assert(file, "cannot open %s", fullname);

        if (!is_sender(comm.rank)) {
            m_log("--------------- %d MSGs ---------------", imsg);
        }
        // for each message size
        size_t max_msg_size = m_msg_size(imsg, param.msg_size, int);
        for (size_t msg_size = 1; msg_size <= max_msg_size; msg_size *= 2) {
            const int idx_size = log10(msg_size) / log10(2.0);
            // get the idx
            int idx = idx_msg * n_size + idx_size;
            // load the results
            const double ti_p2p = m_run_time(p2p_time.avg);
            const double ci_p2p = m_run_time(p2p_time.ci);
            const double ti_put = m_run_time(put_time.avg);
            const double ci_put = m_run_time(put_time.ci);
            const double ti_pgpu = m_run_time(pgpu_time.avg);
            const double ci_pgpu = m_run_time(pgpu_time.ci);
            const double ti_p2pf = m_run_time(p2pf_time.avg);
            const double ci_p2pf = m_run_time(p2pf_time.ci);
            const double ti_fast = m_run_time(pfast_time.avg);
            const double ci_fast = m_run_time(pfast_time.ci);
            const double ti_p2pgpu = m_run_time(p2pgpu_time.avg);
            const double ci_p2pgpu = m_run_time(p2pgpu_time.ci);
            if (!is_sender(comm.rank)) {
                m_log(
                    "time/msg (%ld B/msg - %d msgs):\n"
                    "\tP2P       = %f +-[%f]\n"
                    "\tPUT       = %f +-[%f] (ratio = %f)\n"
                    "\tP2P TRIGR = %f +-[%f] (ratio = %f)\n"
                    "\tPUT TRIGR = %f +-[%f] (ratio = %f)\n"
                    "\tPUT FAST  = %f +-[%f] (ratio = %f)\n"
                    "\tP2P FAST  = %f +-[%f] (ratio = %f)\n",
                    msg_size * sizeof(int), imsg, ti_p2p, ci_p2p, ti_put, ci_put, ti_put / ti_p2p,
                    ti_pgpu, ci_pgpu, ti_pgpu / ti_p2p, ti_p2pgpu, ci_p2pgpu, ti_p2pgpu / ti_p2p,
                    ti_fast, ci_fast, ti_fast / ti_p2p, ti_p2pf, ci_p2pf, ti_p2pf / ti_p2p);
            }
            // write to csv
            fprintf(file, "%ld,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", msg_size * sizeof(int),
                    ti_p2p, ti_put, ti_pgpu, ti_fast, ti_p2pf, ti_p2pgpu, ci_p2p, ci_put, ci_pgpu,
                    ci_fast, ci_p2pf, ci_p2pgpu);
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
        snprintf(fullname, 128, "%s/r%d_size%ld_%s.txt", foldr_name, comm.rank, msg_size,
                 ofi_name(&comm));
        // get a unique FID for the results
        FILE* file = fopen(fullname, "w+");
        m_assert(file, "cannot open %s", fullname);

        // for each message size
        for (int imsg = 1; imsg <= param.n_msg; imsg *= 2) {
            size_t max_msg_size = m_msg_size(imsg, param.msg_size, int);
            if (imsg * msg_size * sizeof(int) > max_msg_size) {
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
            const double ti_pgpu = m_run_time(pgpu_time.avg);
            const double ci_pgpu = m_run_time(pgpu_time.ci);
            const double ti_p2pf = m_run_time(p2pf_time.avg);
            const double ci_p2pf = m_run_time(p2pf_time.ci);
            const double ti_fast = m_run_time(pfast_time.avg);
            const double ci_fast = m_run_time(pfast_time.ci);
            const double ti_p2pgpu = m_run_time(p2pgpu_time.avg);
            const double ci_p2pgpu = m_run_time(p2pgpu_time.ci);
            // write to csv
            fprintf(file, "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", imsg, ti_p2p, ti_put, ti_pgpu,
                    ti_fast, ti_p2pf, ti_p2pgpu, ci_p2p, ci_put, ci_pgpu, ci_fast, ci_p2pf,
                    ci_p2pgpu);
            // bump the index
            idx++;
        }
        fclose(file);
    }
    if (p2p_time.avg) free(p2p_time.avg);
    if (p2p_time.ci) free(p2p_time.ci);
    if (p2pf_time.avg) free(p2pf_time.avg);
    if (p2pf_time.ci) free(p2pf_time.ci);
    if (put_time.avg) free(put_time.avg);
    if (put_time.ci) free(put_time.ci);
    if (pgpu_time.avg) free(pgpu_time.avg);
    if (pgpu_time.ci) free(pgpu_time.ci);
    if (pfast_time.avg) free(pfast_time.avg);
    if (pfast_time.ci) free(pfast_time.ci);

    m_rmem_call(ofi_finalize(&comm));
    return EXIT_SUCCESS;
}
