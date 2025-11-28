#ifndef HELPER_H
#define HELPER_H

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(status, cmd)                                                                         \
    {                                                                                              \
        if ((status) != CL_SUCCESS)                                                                \
        {                                                                                          \
            printf("%s failed (%d)\n", cmd, status);                                               \
            exit(-1);                                                                              \
        }                                                                                          \
    }

// This function reads in a text file and stores it as a char pointer
char *read_source(char *kernel_path);

void init_cl(cl_device_id *device, cl_context *context, cl_program *program);

float *read_filter(const char *filename, int *filter_width);
#endif
