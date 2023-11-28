#include "rmem_qlist.h"
#include "gpu.h"
#include "rmem.h"

#define m_max_n_trigr   (6144)
#define m_gpu_page_size (1 << 16)

/**
 * @brief find the next closest length multiple of the page_size
 */
static size_t qlist_ttl_n_trigr() {
    m_assert(!(m_gpu_page_size % sizeof(uint64_t)), "the gpu page should be a multiple of 8");
    const size_t len = m_max_n_trigr * sizeof(uint64_t);
    const size_t ttl_len = (len + m_gpu_page_size - 1) / m_gpu_page_size;
    return (ttl_len * m_gpu_page_size) / sizeof(uint64_t);
} static size_t qlist_size_bitmap(const size_t n_trigr) {
    return (n_trigr / 8 + ((n_trigr % 8) > 0));
}

static uint8_t mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

// init the trigr with the value of the pointer, as stored in the list
static void trigr_init(rmem_lnode_t* ptr, rmem_trigr_ptr val) {
    // a trigr is ready if the pointer value is odd
    uint64_t uptr = (uint64_t)ptr;
    m_assert((uptr % 2) == 0, "an odd pointer = %p cannot be transformed into a trigger", ptr);
    *val = uptr;
}

/**
 * @brief read the trigger array at idx and return if the trigr is marked ready
 *
 * @param idx: increment the index counter till one of the task is ready, or the next multiple of 8
 * is reached (which corresponds to the length of one element in the bitmap)
 *
 * @returns NULL if no task is ready (therefore, idx %8 ==0), or the pointer to the node that is
 * ready
 */
static rmem_lnode_t* trigr_isready(rmem_trigr_ptr array, const size_t len, uint8_t* bitmap,
                                   int* idx) {
    // get the index inside the uint8_t and the index of the 8 bit
    uint8_t ibit = (*idx) % 8;
    const uint8_t bit = bitmap[(*idx) / 8];
    // total number of bit to read = index in the bit + the number of nodes left to read
    const int n_bit_left = len - (*idx);
    const int n_bit_to_read = m_min(8, ibit + n_bit_left);
    // read those bits
    while (ibit < n_bit_to_read) {
        // test the current bitmap value
        if (bit & mask[ibit]) {
            if (array[*idx] % 2) {
                // inactivate the bitmap and return the node address
                bitmap[(*idx) / 8] &= (~mask[ibit]);
                // inactive the triggering, to avoid retrigering in case of requeueing
                array[*idx] -= 1;
                return (rmem_lnode_t*)(array[*idx]);
            }
        }
        // increment the counter if not ready and keep going while we have the 8 bits in memory
        *idx += 1;
        ibit += 1;
    }
    return NULL;
}
/** 
 * @brief create a list-based multiple producer, single consumer queue
 */
void rmem_lmpsc_create(rmem_lmpsc_t* q) {
    m_countr_init(&q->ongoing);
    m_countr_init(&q->list_count);
    // allocate the bitmap to track completion of the requests, pad if necessary
    const int n_trigr = qlist_ttl_n_trigr();
    q->list_bm = calloc(qlist_size_bitmap(n_trigr), 1);
    if (M_HAVE_GPU) {
        m_gpu_call(
            gpuHostAlloc((void**)&q->h_trigr_list, n_trigr * sizeof(uint64_t), gpuHostAllocMapped));
        m_gpu_call(gpuHostGetDevicePointer((void**)&q->d_trigr_list, (void*)q->h_trigr_list, 0));
    } else {
        q->h_trigr_list = m_malloc(m_gpu_page_size);
        q->d_trigr_list = q->h_trigr_list;
    }
    // init the reset lock
    pthread_mutex_init(&q->reset, NULL);
    // reset the bitmap and the counter
    rmem_lmpsc_reset(q);
}
/** 
 * @brief destroys a list-based multiple producer, single consumer queue
 */
void rmem_lmpsc_destroy(rmem_lmpsc_t* q) {
    m_assert(m_countr_load(&q->ongoing) == 0, "cannot have ongoing operations on the list");
    pthread_mutex_destroy(&q->reset);
    // free the bitmap
    free(q->list_bm);
    // free the device/host memory
    if (M_HAVE_GPU) {
        m_gpu_call(gpuFreeHost((void*)q->h_trigr_list));
    } else {
        free((void*)q->h_trigr_list);
    }
}
void rmem_lmpsc_done(rmem_lmpsc_t* q, rmem_lnode_t* elem) {
    m_assert(m_countr_load(&q->ongoing) > 0, "the number of ongoing op = %d cannot be 0",
             m_countr_load(&q->ongoing));
    m_countr_fetch_add(&q->ongoing, -1);
}
/**
 * @brief reset the list memory, all operations must have completed
 */
static void rmem_lmpsc_requeue(rmem_lmpsc_t* q) {
    m_assert(m_countr_load(&q->ongoing) == 0, "cannot have ongoing operations on the list");
    const int n_trigr = qlist_ttl_n_trigr();
    for (int i = 0; i < qlist_size_bitmap(n_trigr); i++) {
        q->list_bm[i] = 0xff;
    }
}
void rmem_lmpsc_reset(rmem_lmpsc_t* q) {
    pthread_mutex_lock(&q->reset);
    // first requeue everything
    rmem_lmpsc_requeue(q);
    m_countr_rel_store(&q->list_count, 0);
    // unlock the list
    pthread_mutex_unlock(&q->reset);
}

/**
 * @brief enqueue an elem to the list (lockfree & thread-safe function)
 *
 * @returns the (device) trigr handle to be used for that element
 *
 */
rmem_trigr_ptr rmem_lmpsc_enq(rmem_lmpsc_t* q, rmem_lnode_t* elem) {
    // notify a new operation arrives to the queue
    m_countr_fetch_add(&q->ongoing, +1);
    // get the current pool counter
    int pool_idx = m_countr_load(&q->list_count);
    do {
        // repeat while we don't have the right pool_idx
        m_assert(pool_idx < qlist_ttl_n_trigr(),
                 "we have reached maximum enqueuing capacity: %d/%ld", pool_idx,
                 qlist_ttl_n_trigr());
        elem->h_ready_ptr = q->h_trigr_list + pool_idx;
        elem->d_ready_ptr = q->d_trigr_list + pool_idx;
        trigr_init(elem, elem->h_ready_ptr);
    } while (m_countr_rr_cas(&q->list_count, &pool_idx, pool_idx + 1));
    m_verb("THREAD: enqueuing opeation %p, trigr = %lld", elem,*elem->d_ready_ptr);

    // return the handle
    return elem->d_ready_ptr;
}

#define m_n_release 4
/**
 * @brief find the next element that is ready to be executed
 *
 * @param idx: returns the id of the NEXT element to be read
 * @param cnt: returns the count value, used to go through locking/unlocking cycles
 */
void rmem_lmpsc_deq_ifready(rmem_lmpsc_t* q, rmem_lnode_t** elem, int* idx, int* cnt) {
    // prevents the reset while reading the list
    if (!(*cnt)) {
        pthread_mutex_lock(&q->reset);
    }
    // we read the list_count value only once. not super optimized for a general case but good for a
    // case where the list is fixed when dequeueing
    int curr_count = m_countr_acq_load(&q->list_count);
    while (*idx < curr_count) {
        // get the number of bits to read: if we have less than 8 bits to read
        rmem_lnode_t* task = trigr_isready(q->h_trigr_list, curr_count, q->list_bm, idx);
        if (task) {
            *idx += 1;  // next time, start at the next node
            *elem = task;
            m_verb("THREAD: dequeuing opeation %p (current length = %d)", *elem, curr_count);
            goto unlock;
        }
    }
    // if we reach this, we haven't foutn any ready operation preset the output
    *idx = 0;
    *elem = NULL;
unlock:
    // increment the counter FIRST, track the number of cycles we have done before unlocking the
    // queue. Unlocking allows for reset
    *cnt = ((*cnt) + 1) % m_n_release;
    // if the next enter we are going to lock the list, then unlock it first
    if (!(*cnt)) {
        pthread_mutex_unlock(&q->reset);
    }
    return;
}
void rmem_lmpsc_test_cancel(rmem_lmpsc_t* q, int* cnt) {
    // if the count is not 0, we first need to unlock and set the count to 0
    if (*cnt) {
        pthread_mutex_unlock(&q->reset);
        *cnt = 0;
    }
    pthread_testcancel();
}
