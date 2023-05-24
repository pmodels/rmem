/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef RMEM_PROFILE_H_
#define RMEM_PROFILE_H_

#include <time.h>
#include "rmem_utils.h"

#define m_rmem_prof_name_len 256


typedef struct{
  char name [m_rmem_prof_name_len];
  struct timespec t0;
  struct timespec t1;
}rmem_prof_t;

static inline void rmem_prof_start(rmem_prof_t* prof){
  clock_gettime(CLOCK_REALTIME,&prof->t0);
}
static inline void rmem_prof_stop(rmem_prof_t* prof){
  clock_gettime(CLOCK_REALTIME,&prof->t1);
  double time = (long) (prof->t1.tv_sec - prof->t0.tv_sec)*1e+6 + (prof->t1.tv_nsec - prof->t0.tv_nsec)*1.0e-3;
  struct timespec acc;
  clock_getres(CLOCK_REALTIME,&acc);
  m_log("time for %s = %.2f [usec] (accuracy = %ld%09ld [nsec])",prof->name,time,(long)acc.tv_sec,acc.tv_nsec);
}

#define m_dfr(begin, end) for (int m_dfr_i = (begin, 0); !m_dfr_i; (m_dfr_i += 1, end))
#define m_rmem_prof(prof) \
    m_dfr(rmem_prof_start(&prof),rmem_prof_stop(&prof))

#endif
