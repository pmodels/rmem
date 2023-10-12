
#include <inttypes.h>
#include <pthread.h>

#include "ofi.h"
#include "rmem_utils.h"

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
    rmem_qmpsc_t* workq = arg;
    int icancel = 0;
    while (1) {
        // try to dequeue an element
        rmem_qnode_t* task;
        rmem_qmpsc_deq_ifready(workq, &task);
        if (task) {
            m_assert(task->ready, "the task is not ready");
            ofi_rma_start_from_task(task);
            // notify the task has been executed
            m_assert(m_countr_load(workq->done) >= 0, "done counter = %d cannot be <=0",
                     m_countr_load(workq->done));
            m_countr_fetch_add(workq->done, -1);
            m_verb("THREAD: new task done, counter is now %d", m_countr_load(workq->done));
        }
        // test if we need to abort, ideally we don't do this
        icancel = (icancel + 1) % N_CANCEL;
        if (!icancel) {
            pthread_testcancel();
        }
    };
}
