#ifndef CYCLE_TIMER_H
#define CYCLE_TIMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

long long current_ticks();
static double seconds_per_tick();
static double current_seconds();

long long current_ticks()
{
    unsigned int a, d;
    asm volatile("rdtsc" : "=a"(a), "=d"(d));
    return (long long)((unsigned long long)a | (unsigned long long)d << 32);
}

// Return the conversion from ticks to seconds.
static double seconds_per_tick()
{
    static int initialized = 0;
    static double seconds_per_tick_val;
    if (initialized)
        return seconds_per_tick_val;
    FILE *fp = fopen("/proc/cpuinfo", "r");
    char input[1024];
    if (!fp)
    {
        fprintf(stderr, "CycleTimer::resetScale failed: couldn't find /proc/cpuinfo.");
        exit(-1);
    }
    // In case we don't find it, e.g. on the N900
    seconds_per_tick_val = 1e-9;
    while (!feof(fp) && fgets(input, 1024, fp))
    {
        // NOTE(boulos): Because reading cpuinfo depends on dynamic
        // frequency scaling it's better to read the @ sign first
        float g_hz, m_hz;
        if (strstr(input, "model name"))
        {
            char *at_sign = strstr(input, "@");
            if (at_sign)
            {
                char *after_at = at_sign + 1;
                char *g_hz_str = strstr(after_at, "GHz");
                char *m_hz_str = strstr(after_at, "MHz");
                if (g_hz_str)
                {
                    *g_hz_str = '\0';
                    if (1 == sscanf(after_at, "%f", &g_hz))
                    {
                        seconds_per_tick_val = 1e-9f / g_hz;
                        break;
                    }
                }
                else if (m_hz_str)
                {
                    *m_hz_str = '\0';
                    if (1 == sscanf(after_at, "%f", &m_hz))
                    {
                        seconds_per_tick_val = 1e-6f / m_hz;
                        break;
                    }
                }
            }
        }
        else if (1 == sscanf(input, "cpu MHz : %f", &m_hz))
        {
            // printf("MHz = %f\n", MHz);
            seconds_per_tick_val = 1e-6f / m_hz;
            break;
        }
    }
    fclose(fp);

    initialized = 1;
    return seconds_per_tick_val;
}

static double current_seconds()
{
    return (double)current_ticks() * seconds_per_tick();
}

#endif // #ifndef CYCLE_TIMER_H
