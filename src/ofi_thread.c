
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
            m_ofi_rma_structgetptr(ofi_progress_t, rma_prog, progress, task);
            m_ofi_rma_structgetptr(void*, msg_desc, msg.desc_local, task);

            struct fi_msg_rma msg = {
                .msg_iov = msg_iov,
                .desc = msg_desc,
                .iov_count = 1,
                .addr = *addr,
                .rma_iov = msg_riov,
                .rma_iov_count = 1,
                .data = 0x0,
                .context = &msg_cq->ctx,
            };
#if (!M_SYNC_RMA_EVENT)
            msg.data |= m_ofi_data_set_rcq;
#endif
#if (M_WRITE_DATA)
            m_ofi_rma_structgetptr(uint64_t, sig_data, sig.data, task);
            msg.data |= *sig_data;
#endif
            m_verb("doing it");
            m_ofi_call_again(fi_writemsg(*ep, &msg, *msg_flags), rma_prog);
#if (!M_WRITE_DATA)
            // signal if needed
            m_ofi_rma_structgetptr(uint64_t, sig_flags, sig.flags, task);
            if ((*sig_flags)) {
                m_ofi_rma_structgetptr(uint64_t, sig_flags, sig.flags, task);
                m_ofi_rma_structgetptr(struct fi_ioc, sig_iov, sig.iov, task);
                m_ofi_rma_structgetptr(struct fi_rma_ioc, sig_riov, sig.riov, task);
                m_ofi_rma_structgetptr(struct fi_context, sig_ctx, sig.cq.ctx, task);
                m_ofi_rma_structgetptr(void *, sig_desc, sig.desc_local, task);
                struct fi_msg_atomic msg = {
                    .msg_iov = sig_iov,
                    .desc = sig_desc,
                    .iov_count = 1,
                    .addr = *addr,
                    .rma_iov = sig_riov,
                    .rma_iov_count = 1,
                    .datatype = FI_INT32,
                    .op = FI_SUM,
                    .data = 0,
                    .context = sig_ctx,
                };
                m_ofi_call_again(fi_atomicmsg(*ep, &msg, *sig_flags),rma_prog);
            }
#endif
            // if we had to get a cq entry and the inject, mark is as done
            if ((*msg_flags) & FI_INJECT && (*msg_flags) & FI_COMPLETION) {
                m_countr_fetch_add(&msg_cq->rqst.busy, -1);
            }
        }
        // test if we need to abort, ideally we don't do this
        pthread_testcancel();
    };
}
