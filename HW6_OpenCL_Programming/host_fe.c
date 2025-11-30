#include "helper.h"
#include "host_fe.h"
#include <stdio.h>
#include <stdlib.h>

// Static cache to avoid re-initialization overhead
static cl_command_queue command_queue = NULL;
static cl_kernel kernel = NULL;
static cl_mem input_mem = NULL;
static cl_mem filter_mem = NULL;
static cl_mem output_mem = NULL;

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    size_t img_bytes = image_height * image_width * sizeof(float);

    // Initialize only once
    if (command_queue == NULL)
    {
        command_queue = clCreateCommandQueueWithProperties(*context, *device, 0, NULL);

        kernel = clCreateKernel(*program, "convolution", NULL);

        // Use COPY_HOST_PTR to transfer data to GPU once
        size_t flt_bytes = filter_width * filter_width * sizeof(float);
        input_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, img_bytes,
                                   input_image, NULL);
        filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flt_bytes,
                                    filter, NULL);
        output_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, img_bytes, NULL, NULL);

        // Set static arguments
        clSetKernelArg(kernel, 0, sizeof(int), (void *)&filter_width);
        clSetKernelArg(kernel, 2, sizeof(int), (void *)&image_height);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem);
        clSetKernelArg(kernel, 3, sizeof(int), (void *)&image_width);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&input_mem);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_mem);
    }

    size_t local_work_size[2] = {64, 4};
    size_t global_work_size[2] = {(image_width + 63) & ~63, (image_height + 3) & ~3};

    // Execute
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0,
                           NULL, NULL);

    // Read Back Result
    clEnqueueReadBuffer(command_queue, output_mem, CL_TRUE, 0, img_bytes, output_image, 0, NULL,
                        NULL);
}
