
#include "ofi.h"
#include <pthread.h>
#include <inttypes.h>
#include "rmem_utils.h"

#define m_ofi_rma_offset(a)       (offsetof(ofi_rma_t, ofi.a) - offsetof(ofi_rma_t, ofi.qnode))
#define m_ofi_rma_structgetptr(T,name, a, task) T* name = (T*)((uint8_t*)task + m_ofi_rma_offset(a));

void* ofi_tthread_main(void* arg) {
    // some pthread shenanigans
    int old;
    int info = PTHREAD_CANCEL_ENABLE;
    m_pthread_call(pthread_setcancelstate(info, &old));
    // loop on the list forever, the main thread is going to kill it
    rmem_qmpsc_t* workq = arg;
    while (1) {
        // try to dequeue an element
        rmem_qnode_t* task;
        rmem_qmpsc_deq_ifready(workq, &task);
        if (task) {
            m_assert(task->ready, "the task is not ready");
            m_ofi_rma_structgetptr(struct fid_ep*, ep, ep, task);
            m_ofi_rma_structgetptr(fi_addr_t, addr, addr, task);
            // msg specific data
            m_ofi_rma_structgetptr(uint64_t, msg_flags, msg.flags, task);
            m_ofi_rma_structgetptr(ofi_cqdata_t, msg_cq, msg.cq, task);
            m_ofi_rma_structgetptr(struct iovec, msg_iov, msg.iov, task);
            m_ofi_rma_structgetptr(struct fi_rma_iov, msg_riov, msg.riov, task);
            m_log("executing the RMA operation");
            struct fi_msg_rma msg = {
                .msg_iov = msg_iov,
                .desc = NULL,
                .iov_count = 1,
                .addr = *addr,
                .rma_iov = msg_riov,
                .rma_iov_count = 1,
                .data = 0x0,
                .context = &msg_cq->ctx,
            };
            m_ofi_call(fi_writemsg(*ep, &msg, *msg_flags));
            // signal if needed
            m_ofi_rma_structgetptr(uint64_t, sig_flags, sig.flags, task);
            if ((*sig_flags)) {
                m_ofi_rma_structgetptr(uint64_t, sig_flags, sig.flags, task);
                m_ofi_rma_structgetptr(struct fi_ioc, sig_iov, sig.iov, task);
                m_ofi_rma_structgetptr(struct fi_rma_ioc, sig_riov, sig.riov, task);
                m_ofi_rma_structgetptr(struct fi_context, sig_ctx, sig.ctx, task);
                struct fi_msg_atomic msg = {
                    .msg_iov = sig_iov,
                    .desc = NULL,
                    .iov_count = 1,
                    .addr = *addr,
                    .rma_iov = sig_riov,
                    .rma_iov_count = 1,
                    .datatype = FI_INT32,
                    .op = FI_SUM,
                    .data = 0,
                    .context = sig_ctx,
                };
                m_ofi_call(fi_atomicmsg(*ep, &msg, *sig_flags));
            }
            // if we had to get a cq entry and the inject, mark is as done
            if ((*msg_flags) & FI_INJECT && (*msg_flags) & FI_COMPLETION) {
                m_ofi_rma_structgetptr(countr_t, completed, completed, task);
                m_countr_fetch_add(completed, 1);
            }
        }
        // test if we need to abort, ideally we don't do this
        pthread_testcancel();
    };
}
