#include <sys/time.h>

#include "wtime.h"

void wtime(double *t)
{
    static long sec = -1;
    struct timeval tv; // NOLINT(misc-include-cleaner): Clang-Tidy prefers bits/types/timeval.h,
                       // which is not the standard location.
    gettimeofday(&tv, (void *)0);
    if (sec < 0)
        sec = tv.tv_sec;
    *t = (double)(tv.tv_sec - sec) + 1.0e-6 * (double)tv.tv_usec;
}
