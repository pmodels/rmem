/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#include "rmem_utils.h"

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
    m_verb("enqueueing task %p",elem);
    rmem_qnode_t *prev_tail = (rmem_qnode_t *)m_atomicptr_exchange(&q->tail, (intptr_t)elem);
    if (prev_tail) {
        // head != tail, update the new next
        m_atomicptr_store(&prev_tail->next, (intptr_t)elem);
    } else {
        // head == tail, update the new head
        m_atomicptr_store(&q->head, (intptr_t)elem);
    }
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
            if (!m_atomicptr_compare_exchange(&q->tail, (intptr_t *)(elem), 0)) {
                // if the tail is not the dequeued element anymore, wait for the new tail asignment
                while (!m_atomicptr_load(&(*elem)->next)) {
                    continue;
                }
                m_atomicptr_copy(&q->head, &(*elem)->next);
            }
        }
    }
}

/**
 * @brief dequeue ANY element if ready, for a Multiple Producers Single Consumer (MPSC) list
 */
void rmem_qmpsc_deq_ifready(rmem_qmpsc_t *q, rmem_qnode_t **elem) {
    // if we re-read the header pointer, then we need to exit
    int stop = 0;
    do {
        // if the next pointer is null, reset to the head of the list
        if (!m_atomicptr_load(&q->curnt)) {
            m_atomicptr_copy(&q->curnt, &q->head);
            m_atomicptr_store(&q->prev, 0);
            stop++;
        }
        // read the current status, dequeue it if it's ready
        *elem = (rmem_qnode_t *)m_atomicptr_load(&q->curnt);
        if (*elem && (*elem)->ready) {
            m_verb("THREAD: element %p is ready", *elem);
            // item must be dequeue, q->prev stays the same
            if (m_atomicptr_load(&(*elem)->next)) {
                //----------------------------------------------------------------------------------
                // dequeued != tail, so it's safe to update
                m_atomicptr_copy(&q->curnt, &(*elem)->next);
                m_verb("THREAD: current is now %p", (void *)m_atomicptr_load(&q->curnt));
                rmem_qnode_t *prev = (rmem_qnode_t *)m_atomicptr_load(&q->prev);
                if (prev) {
                    m_atomicptr_copy(&prev->next, &(*elem)->next);
                    m_verb("THREAD: update the link from the prev");
                } else {
                    // if we remove the head, get a new head
                    m_atomicptr_copy(&q->head, &(*elem)->next);
                    m_verb("THREAD: head is now %p", (void *)m_atomicptr_load(&q->head));
                }
            } else {
                //----------------------------------------------------------------------------------
                // dequeued = tail, the current is now NULL
                m_atomicptr_store(&q->curnt, 0);
                // if dequeued is the head, update the head
                m_atomicptr_compare_exchange(&q->head, (intptr_t *)(elem),0);
                // try to update the tail, unless is has been changed
                if (!m_atomicptr_compare_exchange(&q->tail, (intptr_t *)(elem), 0)) {
                    // if someone has changed the tail, wait for the update to be done
                    while (!m_atomicptr_load(&(*elem)->next)) {
                        continue;
                    }
                    // the tail is now modified, it's safe to mess around
                    // update the current to the new first 
                    m_atomicptr_copy(&q->curnt, &(*elem)->next);
                    // if dequeued is the head, update the new head
                    m_atomicptr_compare_copy(&q->head, (intptr_t *)(elem), &(*elem)->next);
                }
                m_verb("THREAD: current is now %p", (void *)m_atomicptr_load(&q->curnt));
            }
            m_verb("returning task %p",*elem);
            break;
        } else {
            // go to the next item
            m_atomicptr_copy(&q->prev, &q->curnt);
            if (*elem) {
                m_atomicptr_copy(&q->curnt, &(*elem)->next);
                // the element found is not ready, overwrite to NULL
                *elem = NULL;
            }
        }
    } while (!stop);
}
