/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "rmem.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/**
 * @brief prints the backtrace history
 *
 * based on
 * - https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
 * - https://gist.github.com/fmela/591333/c64f4eb86037bb237862a8283df70cdfc25f01d3
 * - https://linux.die.net/man/3/dladdr
 *
 * The symbols generated might be weird if the demangler doesn't work correctly
 * if you don't see any function name, add the option `-rdynamic` to the linker's flag
 *
 */
void PrintBackTrace() {
    //--------------------------------------------------------------------------
#if (1 == M_BACKTRACE)
    // get the addresses of the function currently in execution
    void *call_stack[M_BACKTRACE_HISTORY];  // array of
    int size = backtrace(call_stack, M_BACKTRACE_HISTORY);

    // transform the pointers into readable symbols
    char **strings;
    strings = backtrace_symbols(call_stack, size);
    if (strings != NULL) {
        m_log("--------------------- CALL STACK ----------------------");
        // we start at 1 to not display this function
        for (int i = 1; i < size; i++) {
            //  display the demangled name if we succeeded, weird name if not
            m_log("%s", strings[i]);
        }
        m_log("-------------------------------------------------------");
    }
    // free the different string with the names
    free(strings);
#endif
    //--------------------------------------------------------------------------
}

/**
 * @brief enqueue a new node at the TAIL for a Multiple Producers Single Consumer (MPSC) list
 *
 */
void rmem_qmpsc_enq(rmem_qmpsc_t *q, rmem_qnode_t *elem) {
    m_verb("enqueueing task %p, value = %d",elem, elem->ready);
    // overwrite the next value
    m_atomicptr_store(&elem->next,0);
    // update the tail
    rmem_qnode_t *prev_tail = (rmem_qnode_t *)m_atomicptr_swap(&q->tail, (intptr_t)elem);
    if (prev_tail) {
        // head != tail, update the new next
        m_atomicptr_store(&prev_tail->next, (intptr_t)elem);
    } else {
        // head == tail, update the new head
        m_atomicptr_store(&q->head, (intptr_t)elem);
    }
    m_verb("enq: current is now %p", (void *)m_atomicptr_load(&q->curnt));
    m_verb("enq: head is now %p", (void *)m_atomicptr_load(&q->head));
    m_verb("enq: tail is now %p", (void *)m_atomicptr_load(&q->tail));
}

/**
 * @brief dequeue the HEAD for a Multiple Producers Single Consumer (MPSC) list
 *
 */
void rmem_qmpsc_deq(rmem_qmpsc_t *q, rmem_qnode_t **elem) {
    // get the current head
    *elem = (rmem_qnode_t *)m_atomicptr_load(&q->head);
    // if the queue is not empty
    if (*elem) {
        if (m_atomicptr_load(&(*elem)->next)) {
            // previous head is NOT empty, use it
            m_atomicptr_copy(&q->head, &(*elem)->next);
        } else {
            // head was empty, maybe someone is updating
            m_atomicptr_store(&q->head, 0);
            if (!m_atomicptr_cas(&q->tail, (intptr_t *)(elem), 0)) {
                // if the tail is not the dequeued element anymore, wait for the new tail asignment
                while (!m_atomicptr_load(&(*elem)->next)) {
                    continue;
                }
                m_atomicptr_copy(&q->head, &(*elem)->next);
            }
        }
    }
}

static void rmem_qmpsc_update_ptr(rmem_qmpsc_t *q, rmem_qnode_t *prev, const atomic_ptr_t *new) {
    // 1. bump the reading ptr
    m_atomicptr_copy(&q->curnt, new);
    // 2. change the previous element
    if (prev) {
        // dequeue != HEAD
        m_atomicptr_copy(&prev->next, new);
    } else {
        // dequeue == HEAD
        m_atomicptr_copy(&q->head, new);
    }
}

/**
 * @brief dequeue ANY element if ready, for a Multiple Producers Single Consumer (MPSC) list
 */
void rmem_qmpsc_deq_ifready(rmem_qmpsc_t *q, rmem_qnode_t **res) {
    // if we re-read the header pointer, then we need to exit
    do {
        // if the next pointer is null, reset to the head of the list
        if (!m_atomicptr_load(&q->curnt)) {
            m_atomicptr_copy(&q->curnt, &q->head);
            m_atomicptr_store(&q->prev, 0);
            *res = NULL;
            break;
        } else {
            // read the current status, dequeue it if it's ready
            const intptr_t elem = m_atomicptr_load(&q->curnt);
            m_assert(elem, "the element must be non-NULL here");
            *res = (rmem_qnode_t *)elem;
            if ((*res)->ready) {
                m_verb("THREAD: elem = %ld, ready? %d", elem, (*res)->ready);
                // read the previous value
                rmem_qnode_t *prev = (rmem_qnode_t *)m_atomicptr_load(&q->prev);
                // item must be dequeue, q->prev stays the same
                if (m_atomicptr_load(&(*res)->next)) {
                    m_verb("THREAD: found a next one");
                    //----------------------------------------------------------------------------------
                    // dequeued != tail, so it's safe to update
                    rmem_qmpsc_update_ptr(q, prev, &(*res)->next);
                } else {
                    // dequeue = TAIL
                    m_verb("THREAD: dequeue the tail");
                    //----------------------------------------------------------------------------------
                    // [1] first pretend that there are no update on the tail
                    atomic_ptr_t null_ptr = {.ptr = 0};
                    rmem_qmpsc_update_ptr(q, prev, &null_ptr);

                    // [2] test if the tail is updated, if so, wait for the update to be over
                    intptr_t expected = elem;
                    if (!m_atomicptr_cas(&q->tail, &expected, q->prev.ptr)) {
                        // the tail has been changed, wait for the next value to be available
                        while (!m_atomicptr_load(&(*res)->next)) {
                            continue;
                        }
                        rmem_qmpsc_update_ptr(q, prev, &(*res)->next);
                    }
                }
                m_verb("THREAD: returning task %ld with value %d", elem, (*res)->ready);
                m_verb("THREAD: current is now %p", (void *)m_atomicptr_load(&q->curnt));
                m_verb("THREAD: head is now %p", (void *)m_atomicptr_load(&q->head));
                m_verb("THREAD: tail is now %p", (void *)m_atomicptr_load(&q->tail));
                break;
            } else {
                // go to the next item
                // if (elem) {
                m_atomicptr_store(&q->prev, elem);
                m_atomicptr_copy(&q->curnt, &(*res)->next);
                // the element found is not ready, overwrite to NULL
                // *res = NULL;
                // }
            }
        }
    } while (1);
}
