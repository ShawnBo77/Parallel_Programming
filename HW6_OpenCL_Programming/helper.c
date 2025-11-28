#include "helper.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This function reads in a text file and stores it as a char pointer
char *read_source(char *kernel_path)
{
    printf("Program file is: %s\n", kernel_path);

    FILE *fp = fopen(kernel_path, "rb");
    if (!fp)
    {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    cl_int status = fseek(fp, 0, SEEK_END);
    if (status != 0)
    {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    long size = ftell(fp);
    if (size < 0)
    {
        printf("Error getting file position\n");
        exit(-1);
    }

    status = fseek(fp, 0, SEEK_SET);
    if (status != 0)
    {
        printf("Error seeking to beginning of file\n");
        exit(-1);
    }

    char *source = malloc(size + 1);
    if (!source)
    {
        printf("Error allocating memory for kernel source\n");
        fclose(fp);
        exit(-1);
    }

    if (source == NULL)
    {
        printf("Error allocating space for the kernel source\n");
        fclose(fp);
        exit(-1);
    }

    int i;
    for (i = 0; i < size + 1; i++)
    {
        source[i] = '\0';
    }

    if (source == NULL)
    {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    if (fread(source, 1, size, fp) != size)
    {
        printf("Error reading kernel source\n");
        free(source);
        fclose(fp);
        exit(-1);
    }
    source[size] = '\0';

    fclose(fp);
    return source;
}

void init_cl(cl_device_id *device, cl_context *context, cl_program *program)
{
    // Set up the OpenCL environment
    cl_int status;

    // Discovery platform
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK(status, "clGetPlatformIDs");

    // Discover device
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
    CHECK(status, "clGetDeviceIDs");

    // Create context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
    *context = clCreateContext(props, 1, device, NULL, NULL, &status);
    CHECK(status, "clCreateContext");

    const char *source = read_source("kernel.cl");

    // Create a program object with source and build it
    *program = clCreateProgramWithSource(*context, 1, &source, NULL, NULL);
    CHECK(status, "clCreateProgramWithSource");
    status = clBuildProgram(*program, 1, device, NULL, NULL, NULL);
    CHECK(status, "clBuildProgram");
}

float *read_filter(const char *filename, int *filter_width)
{
    printf("Reading filter data from %s\n", filename);

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        printf("Could not open filter file\n");
        exit(-1);
    }

    if (fscanf(fp, "%d", filter_width) <= 0)
    {
        printf("Error reading filter width\n");
        fclose(fp);
        exit(-1);
    }

    float *filter = malloc(*filter_width * *filter_width * sizeof(float));
    if (!filter)
    {
        printf("Error allocating memory for filter\n");
        fclose(fp);
        exit(-1);
    }

    for (int i = 0; i < *filter_width * *filter_width; i++)
    {
        float tmp;
        if (fscanf(fp, "%f", &tmp) <= 0)
        {
            printf("Error reading filter data\n");
            free(filter);
            fclose(fp);
            exit(-1);
        }
        filter[i] = tmp;
    }

    printf("Filter width: %d\n", *filter_width);

    fclose(fp);
    return filter;
}
