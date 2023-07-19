/*
 * Copyright (C) by Argonne National Laboratory
 *	See COPYRIGHT in top-level directory
 */
#ifndef RMEM_PROFILE_H_
#define RMEM_PROFILE_H_

#include <limits.h>
#include <math.h>
#include <time.h>

#include "rmem_utils.h"

#define m_rmem_prof_name_len 256

#define m_rmem_nu_len 14
static int t_student_nu[m_rmem_nu_len] = {0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 500};
static double t_student_t[m_rmem_nu_len] = {0.0,   6.314, 2.920, 2.353, 2.132, 2.015, 1.895,
                                            1.812, 1.753, 1.725, 1.697, 1.676, 1.660, 1.645};

static inline double t_nu_interp(const int nu) {
    m_assert(nu >= 0, "the nu param = %d must be positive", nu);
    //--------------------------------------------------------------------------
    if (nu == 0) {
        // easy, it's 0
        return 0.0;
    } else if (nu <= 5) {
        // we have an exact entry
        return t_student_t[nu];
    } else if (nu >= t_student_nu[m_rmem_nu_len-1]) {
        // we are too big, it's like a normal distribution
        return t_student_t[m_rmem_nu_len-1];
    } else {
        int i = 5;
        while (t_student_nu[i] < nu) {
            i++;
        }
        const int nu_low = t_student_nu[i - 1];
        const int nu_up = t_student_nu[i];
        const double t_low = t_student_t[i - 1];
        const double t_up = t_student_t[i];
        // find the right point
        return t_low + (t_up - t_low) / (nu_up - nu_low) * (nu - nu_low);
    }
    //--------------------------------------------------------------------------
}

static inline void rmem_get_ci(const int n_data, double* data, double* avg, double* ci) {
    const double t_nu_val = t_nu_interp(n_data);
    *avg = 0.0;
    for (int i = 0; i < n_data; ++i) {
        *avg += data[i] / n_data;
    }
    double std = 0.0;
    for (int i = 0; i < n_data; ++i) {
        std += pow(data[i] - *avg, 2);
    }
    const double s = sqrt(std / (n_data - 1));
    *ci = s * t_nu_val * sqrt(1.0 / n_data);
}

typedef struct{
  char name [m_rmem_prof_name_len];
  struct timespec t0;
  struct timespec t1;
}rmem_prof_t;

static inline void rmem_prof_start(rmem_prof_t* prof){
  clock_gettime(CLOCK_REALTIME,&prof->t0);
}
static inline void rmem_prof_stop(rmem_prof_t* prof, double* time){
  clock_gettime(CLOCK_REALTIME,&prof->t1);
  *time = (long) (prof->t1.tv_sec - prof->t0.tv_sec)*1e+6 + (prof->t1.tv_nsec - prof->t0.tv_nsec)*1.0e-3;
  struct timespec acc;
  clock_getres(CLOCK_REALTIME,&acc);
  //m_log("time for %s = %.2f [usec] (accuracy = %ld%09ld [nsec])",prof->name,*time,(long)acc.tv_sec,acc.tv_nsec);
}

#define m_dfr(begin, end) for (int m_dfr_i = (begin, 0); !m_dfr_i; (m_dfr_i += 1, end))
#define m_rmem_prof(prof,time) \
    m_dfr(rmem_prof_start(&prof),rmem_prof_stop(&prof,&time))

#endif
