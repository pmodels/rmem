#ifndef RMEM_TRIGR_H_
#define RMEM_TRIGR_H_
#include <stdint.h>
#include <stdlib.h>
#include "rmem_utils.h"


// define a trigr used in the queueing system to start an operation
// the operation is ready if the value is odd
typedef volatile uint64_t* rmem_trigr_ptr;

/**
 * @brief trigers the operation
 *
 * note: must remain a macro to facilitate GPU/CPU usage
 */
#define m_rmem_trigger(trigr)                        \
    do {                                             \
        rmem_trigr_ptr rmem_trigger_trigr = (trigr); \
        (*rmem_trigger_trigr)++;                     \
    } while (0)

#endif
