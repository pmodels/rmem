
#include <inttypes.h>
#include <pthread.h>

#include "ofi.h"
#include "rmem_qlist.h"
#include "rmem_utils.h"
#include "ofi_rma_sync_tools.h"

#define N_CANCEL            1000
#define m_ofi_rma_offset(a) (offsetof(ofi_rma_t, ofi.a) - offsetof(ofi_rma_t, ofi.qnode))
#define m_ofi_rma_structgetptr(T, name, a, task) \
    T* name = (T*)((uint8_t*)task + m_ofi_rma_offset(a));

void* ofi_tthread_main(void* arg) {
    // some pthread shenanigans
    int old;
    int info = PTHREAD_CANCEL_ENABLE;
    m_pthread_call(pthread_setcancelstate(info, &old));
    // loop on the list forever, the main thread is going to kill it
    const rmem_thread_arg_t* thread_arg = (rmem_thread_arg_t*)arg;
    rmem_lmpsc_t* workq = thread_arg->workq;
    // rmem_qmpsc_t* workq = thread_arg->workq;
    ofi_progress_t progress = {
        .cq = NULL,
        .xctx.epoch_ptr = thread_arg->xctx.epoch_ptr,
    };

    int icancel = 0;     // count the number of iterations between cancelation test
    int idequeue = 0;    // count the number of iterations between the lock/unlock
    int search_idx = 0;  // index on which item to search next
    while (1) {
        // try to dequeue an element
        rmem_lnode_t* task;
        rmem_lmpsc_deq_ifready(workq, &task, &search_idx, &idequeue);
        if (task) {
            m_assert(task->h_ready_ptr[0], "the task is not ready");
            // choose the right action to take
            switch (task->kind) {
                case (LNODE_KIND_RMA): {
                    ofi_rma_start_from_task(task);
                } break;
                case (LNODE_KIND_COMPL): {
                    // the task is a node but the address is the same for a complete_ack_t
                    ofi_rmem_issue_dtc((rmem_complete_ack_t*)task);
                } break;
                default:
                    m_assert(false, "unknown kind argument");
            }
            rmem_lmpsc_done(workq, task);
            m_verb("THREAD: new task done, counter is now %d", m_countr_load(&workq->ongoing));
        } else if (m_countr_load(&workq->ongoing)) {
            int fail = pthread_mutex_trylock(thread_arg->do_progress);
            if (!fail) {
                // make progress if allowed AND we have outgoing operations
                for (int i = 0; i < thread_arg->n_tx + 1; ++i) {
                    progress.cq = thread_arg->data_trx[i].cq;
                    m_ofi_call(ofi_progress(&progress));
                }
                pthread_mutex_unlock(thread_arg->do_progress);
            }
        }
        // test if we need to abort, ideally we don't do this
        icancel = (icancel + 1) % N_CANCEL;
        if (!icancel) {
            rmem_lmpsc_test_cancel(workq, &idequeue);
        }
    }
}
